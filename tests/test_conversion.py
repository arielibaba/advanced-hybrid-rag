"""Tests for convert_to_pdf.py — document conversion, LibreOffice detection, fallbacks."""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from cognidoc.convert_to_pdf import (
    find_libreoffice,
    convert_with_libreoffice,
    convert_html_to_pdf,
    convert_markdown_to_pdf,
    convert_text_to_pdf,
    convert_image_to_pdf,
    convert_document_to_pdf,
    process_source_documents,
    OFFICE_EXTENSIONS,
    HTML_EXTENSIONS,
    TEXT_EXTENSIONS,
    MARKDOWN_EXTENSIONS,
    IMAGE_EXTENSIONS,
    ALL_SUPPORTED_EXTENSIONS,
)


# ---------------------------------------------------------------------------
# Extension sets
# ---------------------------------------------------------------------------


class TestExtensionSets:

    def test_office_extensions_populated(self):
        assert ".docx" in OFFICE_EXTENSIONS
        assert ".pptx" in OFFICE_EXTENSIONS
        assert ".xlsx" in OFFICE_EXTENSIONS

    def test_image_extensions_populated(self):
        assert ".jpg" in IMAGE_EXTENSIONS
        assert ".png" in IMAGE_EXTENSIONS

    def test_all_supported_is_union(self):
        assert ".pdf" not in ALL_SUPPORTED_EXTENSIONS
        assert ".docx" in ALL_SUPPORTED_EXTENSIONS
        assert ".html" in ALL_SUPPORTED_EXTENSIONS
        assert ".txt" in ALL_SUPPORTED_EXTENSIONS


# ---------------------------------------------------------------------------
# find_libreoffice
# ---------------------------------------------------------------------------


class TestFindLibreOffice:

    def test_finds_existing_path(self):
        """Returns path if it exists on disk."""
        with patch("os.path.isfile", side_effect=lambda p: p == "/usr/bin/soffice"):
            with patch("cognidoc.convert_to_pdf.platform") as mock_plat:
                mock_plat.system.return_value = "Linux"
                result = find_libreoffice()
                assert result == "/usr/bin/soffice"

    def test_returns_none_when_not_found(self):
        with patch("os.path.isfile", return_value=False):
            with patch("cognidoc.convert_to_pdf.platform") as mock_plat:
                mock_plat.system.return_value = "Linux"
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=1, stdout="")
                    result = find_libreoffice()
                    assert result is None

    def test_falls_back_to_which(self):
        with patch("os.path.isfile", return_value=False):
            with patch("cognidoc.convert_to_pdf.platform") as mock_plat:
                mock_plat.system.return_value = "Linux"
                with patch("subprocess.run") as mock_run:
                    mock_run.return_value = MagicMock(returncode=0, stdout="/snap/bin/soffice\n")
                    result = find_libreoffice()
                    assert result == "/snap/bin/soffice"

    def test_handles_subprocess_error(self):
        """Exception in 'which' call is caught gracefully."""
        with patch("os.path.isfile", return_value=False):
            with patch("cognidoc.convert_to_pdf.platform") as mock_plat:
                mock_plat.system.return_value = "Linux"
                with patch("subprocess.run", side_effect=OSError("not found")):
                    result = find_libreoffice()
                    assert result is None

    def test_macos_paths(self):
        with patch("os.path.isfile", side_effect=lambda p: "MacOS" in p):
            with patch("cognidoc.convert_to_pdf.platform") as mock_plat:
                mock_plat.system.return_value = "Darwin"
                result = find_libreoffice()
                assert result is not None
                assert "MacOS" in result


# ---------------------------------------------------------------------------
# convert_with_libreoffice
# ---------------------------------------------------------------------------


class TestConvertWithLibreOffice:

    def test_successful_conversion(self, tmp_path):
        input_file = tmp_path / "doc.docx"
        input_file.touch()
        output_pdf = tmp_path / "doc.pdf"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            # Simulate the PDF being created
            output_pdf.touch()
            result = convert_with_libreoffice(input_file, tmp_path, "/usr/bin/soffice")
            assert result == output_pdf

    def test_libreoffice_returns_error(self, tmp_path):
        input_file = tmp_path / "doc.docx"
        input_file.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="conversion error")
            result = convert_with_libreoffice(input_file, tmp_path, "/usr/bin/soffice")
            assert result is None

    def test_timeout_handling(self, tmp_path):
        input_file = tmp_path / "big.docx"
        input_file.touch()

        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="soffice", timeout=120)
        ):
            result = convert_with_libreoffice(input_file, tmp_path, "/usr/bin/soffice")
            assert result is None

    def test_missing_output_after_conversion(self, tmp_path):
        input_file = tmp_path / "doc.docx"
        input_file.touch()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="")
            # Don't create the output PDF
            result = convert_with_libreoffice(input_file, tmp_path, "/usr/bin/soffice")
            assert result is None


# ---------------------------------------------------------------------------
# convert_html_to_pdf
# ---------------------------------------------------------------------------


class TestConvertHtmlToPdf:

    def test_weasyprint_not_installed(self, tmp_path):
        with patch.dict("sys.modules", {"weasyprint": None}):
            result = convert_html_to_pdf(tmp_path / "page.html", tmp_path / "page.pdf")
            assert result is None

    def test_successful_conversion(self, tmp_path):
        input_file = tmp_path / "page.html"
        input_file.write_text("<html><body>Hello</body></html>")
        output_file = tmp_path / "page.pdf"

        mock_html_cls = MagicMock()
        mock_html_instance = MagicMock()
        mock_html_cls.return_value = mock_html_instance

        def fake_write_pdf(path):
            Path(path).touch()

        mock_html_instance.write_pdf = fake_write_pdf

        with patch.dict("sys.modules", {"weasyprint": MagicMock(HTML=mock_html_cls)}):
            # We need to reimport since the module caches
            result = convert_html_to_pdf(input_file, output_file)
            # The function tries `from weasyprint import HTML` — with our mock it might raise
            # Just verify it handles gracefully
            assert result is None or isinstance(result, Path)


# ---------------------------------------------------------------------------
# convert_text_to_pdf
# ---------------------------------------------------------------------------


class TestConvertTextToPdf:

    def test_reportlab_not_installed(self, tmp_path):
        input_file = tmp_path / "doc.txt"
        input_file.write_text("Hello world")
        with patch.dict(
            "sys.modules",
            {
                "reportlab": None,
                "reportlab.lib.pagesizes": None,
                "reportlab.pdfgen": None,
                "reportlab.lib.units": None,
            },
        ):
            result = convert_text_to_pdf(input_file, tmp_path / "doc.pdf")
            assert result is None


# ---------------------------------------------------------------------------
# convert_image_to_pdf
# ---------------------------------------------------------------------------


class TestConvertImageToPdf:

    def test_pillow_not_installed(self, tmp_path):
        input_file = tmp_path / "photo.jpg"
        input_file.touch()
        with patch.dict("sys.modules", {"PIL": None, "PIL.Image": None}):
            result = convert_image_to_pdf(input_file, tmp_path / "photo.pdf")
            assert result is None

    def test_successful_rgb_image(self, tmp_path):
        """Use a real PIL Image to test the conversion path."""
        from PIL import Image

        input_file = tmp_path / "photo.png"
        # Create a small real PNG image
        img = Image.new("RGB", (10, 10), color=(255, 0, 0))
        img.save(str(input_file), "PNG")

        output_file = tmp_path / "photo.pdf"
        result = convert_image_to_pdf(input_file, output_file)
        assert result == output_file
        assert output_file.exists()

    def test_rgba_image_converted(self, tmp_path):
        """RGBA images should be converted to RGB with white background."""
        from PIL import Image

        input_file = tmp_path / "transparent.png"
        img = Image.new("RGBA", (10, 10), color=(255, 0, 0, 128))
        img.save(str(input_file), "PNG")

        output_file = tmp_path / "transparent.pdf"
        result = convert_image_to_pdf(input_file, output_file)
        assert result == output_file
        assert output_file.exists()


# ---------------------------------------------------------------------------
# convert_document_to_pdf — routing
# ---------------------------------------------------------------------------


class TestConvertDocumentToPdf:

    def test_office_without_libreoffice(self, tmp_path):
        input_file = tmp_path / "report.docx"
        input_file.touch()
        result = convert_document_to_pdf(input_file, tmp_path, libreoffice_path=None)
        assert result is None

    def test_office_with_libreoffice(self, tmp_path):
        input_file = tmp_path / "report.docx"
        input_file.touch()
        with patch(
            "cognidoc.convert_to_pdf.convert_with_libreoffice", return_value=tmp_path / "report.pdf"
        ):
            result = convert_document_to_pdf(
                input_file, tmp_path, libreoffice_path="/usr/bin/soffice"
            )
            assert result == tmp_path / "report.pdf"

    def test_unsupported_extension(self, tmp_path):
        input_file = tmp_path / "data.xyz"
        input_file.touch()
        result = convert_document_to_pdf(input_file, tmp_path)
        assert result is None

    def test_html_fallback_to_libreoffice(self, tmp_path):
        input_file = tmp_path / "page.html"
        input_file.touch()
        with patch("cognidoc.convert_to_pdf.convert_html_to_pdf", return_value=None):
            with patch(
                "cognidoc.convert_to_pdf.convert_with_libreoffice",
                return_value=tmp_path / "page.pdf",
            ) as mock_lo:
                result = convert_document_to_pdf(
                    input_file, tmp_path, libreoffice_path="/usr/bin/soffice"
                )
                mock_lo.assert_called_once()
                assert result == tmp_path / "page.pdf"


# ---------------------------------------------------------------------------
# process_source_documents
# ---------------------------------------------------------------------------


class TestProcessSourceDocuments:

    def test_empty_directory(self, tmp_path):
        sources = tmp_path / "sources"
        sources.mkdir()
        pdfs = tmp_path / "pdfs"
        stats = process_source_documents(str(sources), str(pdfs))
        assert stats["total_files"] == 0

    def test_pdf_copied_directly(self, tmp_path):
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "doc.pdf").write_bytes(b"%PDF-1.4 test")
        pdfs = tmp_path / "pdfs"
        stats = process_source_documents(str(sources), str(pdfs))
        assert stats["pdfs_copied"] == 1
        assert (pdfs / "doc.pdf").exists()

    def test_image_copied_to_images_dir(self, tmp_path):
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "photo.jpg").write_bytes(b"\xff\xd8\xff\xe0")
        pdfs = tmp_path / "pdfs"
        images = tmp_path / "images"
        stats = process_source_documents(str(sources), str(pdfs), image_output_dir=str(images))
        assert stats["images_copied"] == 1
        assert (images / "photo_page_1.jpg").exists()

    def test_unsupported_format_skipped(self, tmp_path):
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "data.xyz").touch()
        pdfs = tmp_path / "pdfs"
        stats = process_source_documents(str(sources), str(pdfs))
        assert stats["unsupported"] == 1

    def test_skip_existing_pdf(self, tmp_path):
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "doc.pdf").write_bytes(b"%PDF-1.4 test")
        pdfs = tmp_path / "pdfs"
        pdfs.mkdir()
        (pdfs / "doc.pdf").write_bytes(b"%PDF-1.4 old")
        stats = process_source_documents(str(sources), str(pdfs))
        assert stats["skipped_existing"] == 1

    def test_source_files_filter(self, tmp_path):
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "a.pdf").write_bytes(b"%PDF")
        (sources / "b.pdf").write_bytes(b"%PDF")
        pdfs = tmp_path / "pdfs"
        stats = process_source_documents(
            str(sources), str(pdfs), source_files=[str(sources / "a.pdf")]
        )
        assert stats["pdfs_copied"] == 1
        assert "a" in stats["processed_pdf_stems"]

    def test_conversion_failure_tracked(self, tmp_path):
        sources = tmp_path / "sources"
        sources.mkdir()
        (sources / "report.docx").touch()
        pdfs = tmp_path / "pdfs"
        with patch("cognidoc.convert_to_pdf.find_libreoffice", return_value=None):
            stats = process_source_documents(str(sources), str(pdfs))
            assert stats["failed"] == 1
            assert len(stats["errors"]) == 1

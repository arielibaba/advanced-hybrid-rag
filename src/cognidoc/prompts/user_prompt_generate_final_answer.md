## Conversation History:
{conversation_history}

## User Question:
{user_question}

## Retrieved Context:
{refined_context}

## Instructions:
Provide a clear, integrated answer addressing all parts of the user's question.
- If there are multiple sub-questions, address each explicitly.
- Do **not** reference the retrieval process or the provided context explicitly.

### Handling Partial Information:
- If the context contains **related information** but does not **fully answer** the specific question:
  1. Share the relevant information that IS available in the documents
  2. Clearly state what aspect of the question cannot be answered from the available documents
  3. Use phrasing like (adapt to user's language):
     - "Les documents contiennent des informations sur [sujet connexe]... Cependant, concernant [aspect spécifique de la question], la base documentaire ne fournit pas d'information explicite."
     - "The documents provide information about [related topic]... However, regarding [specific aspect], the document base does not provide explicit information."
  4. **NEVER invent or extrapolate** information not present in the context

### Only if NO relevant information at all:
- If there is truly **nothing relevant** in the context (not even related topics), respond in the user's language:
  - French: **"Je n'ai pas trouvé d'informations pertinentes dans la base documentaire pour répondre à cette question."**
  - English: **"I could not find relevant information in the document base to answer this question."**
  - Spanish: **"No he encontrado información relevante en la base documental para responder a esta pregunta."**
  - German: **"Ich habe keine relevanten Informationen in der Dokumentenbasis gefunden, um diese Frage zu beantworten."**

- CRITICAL: Deliver your ENTIRE response in the SAME LANGUAGE as the user's question. If the user asks in French, respond entirely in French. If in English, respond entirely in English. If in Spanish, respond entirely in Spanish. If in German, respond entirely in German.
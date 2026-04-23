---
name: example-skill
description: Demonstrates the skill format. Use when the user asks for a structured summary/report of any topic, or wants to count words or reverse text.
---

# Example Skill: Structured Summary & Text Utilities

## When to use this skill
Use this skill when the user asks you to summarize, report on, or give an overview of any topic in a structured way, or when they ask to count words or reverse text.

## Instructions

### Structured summary
1. Identify the main topic from the user's request.
2. Produce a response with the following sections:
   - **Overview**: 2-3 sentence summary of the topic.
   - **Key Points**: 3-5 bullet points covering the most important aspects.
   - **Limitations / Caveats**: anything the user should be aware of.
3. Keep the tone neutral and concise.
4. If MCP tools are active and relevant, use them to gather information first.

### Text utilities
Call `skill__example-skill__word_count` or `skill__example-skill__reverse_text` directly.
Wait for the result, then answer the user in plain prose.

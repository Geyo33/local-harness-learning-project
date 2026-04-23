---
name: calculator
description: Instructions to perform flawless arithmetic operations (add, subtract, multiply, divide). Use when the user asks to calculate, compute, or do math.
---

# Calculator Skill

## When to use this skill
Use this skill whenever the user asks you to perform an arithmetic calculation.

## Available functions

| Function   | Arguments       | Description         |
|------------|-----------------|---------------------|
| add        | a, b (numbers)  | Returns a + b       |
| subtract   | a, b (numbers)  | Returns a - b       |
| multiply   | a, b (numbers)  | Returns a * b       |
| divide     | a, b (numbers)  | Returns a / b       |

## Instructions

1. Identify the operation and operands from the user's request.
2. Call the appropriate function tool (e.g. `skill__calculator__add`).
3. Wait for the result.
4. Answer the user in plain prose (e.g. "42 × 7 = 294").

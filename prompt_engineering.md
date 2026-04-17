# Prompt Engineering Reference

Internal reference for writing system prompts in this project. Compiled from Brex's production prompt engineering guide and leaked Cursor/Devin AI system prompts.

---

## Structure

### Use XML tags to scope sections

Models treat XML tags as bounded rule sets. Each section gets its own tag so the model can reference rules contextually without bleeding across concerns.

```
<identity>
Role definition and scope boundaries.
</identity>

<rules>
Behavioral constraints.
</rules>

<output_format>
Schema and format requirements.
</output_format>

<examples>
Good/bad pairs.
</examples>
```

Cursor uses: `<communication>`, `<tool_calling>`, `<making_code_changes>`, `<citing_code>`.
Devin uses: `<think>`, `<shell>`, `<open_file>` as action delimiters.

### Information ordering matters

1. **Identity/role** — first. Establishes scope before any rules.
2. **Context/data** — what the model needs to know to do its job.
3. **Rules/constraints** — operational behavior.
4. **Output format** — schema, types, examples.
5. **Critical constraints repeated at end** — resilience against attention collapse. Models lose focus on instructions that appear only at the start of long prompts. Restating the 2-3 most important rules at the very end ensures they're attended to.

### Embed data in the right format

| Data shape | Best format |
|---|---|
| Single object, few fields | Bulleted list |
| Multiple same-schema items | Markdown table |
| Sparse/irregular fields | JSON |
| Long documents | Triple-backtick blocks |
| Related tables | Relational tables with FK refs |

---

## Constraints

### Emphasis hierarchy

ALL CAPS > `[CRITICAL]` markers > **bold** > plain text.

Critical rules should use multiple emphasis levels:
```
[CRITICAL] NEVER include raw sensor values in the narration.
```

### Negative constraints dominate

"NEVER do X" and "DO NOT do Y" are more reliable than "try to avoid X". Both Cursor and Devin lean heavily on explicit prohibitions:
- "NEVER refer to tool names when speaking to the user"
- "DO NOT omit spans of pre-existing code"
- "Never force push"
- "Never use grep or find to search"

### Repeat critical rules in different phrasings

Brex: "Reiterate important behavioral rules near prompt end."
Cursor: Critical rules appear 3-5 times across sections in reformulated phrasings. This creates resilience against attention collapse while appearing non-repetitive.

Example: "use tool calls instead of asking" appears in Cursor's `<tool_calling>`, `<maximize_context_understanding>`, and `<making_code_changes>` sections — same rule, three contexts.

### Scope constraints with conditionals

Instead of absolute rules that break in edge cases, use conditional scope:
- "IF creating a file from scratch, write the complete implementation"
- "When encountering difficulties, try alternative approaches before asking"
- "If you fail to edit a file, read the file again first"

---

## Output formatting

### Show the exact JSON schema with type annotations

Don't just describe the output — show it with types and comments:
```
{
  "narration": string,        // [CRITICAL] one sentence
  "escalate": boolean,        // true ONLY for occupancy changes
  "escalate_reason": string   // brief reason
}
```

This is more reliable than "return a JSON object with a narration field."

### Separate thinking from output

If the model needs to reason internally, use delimited sections:
```
"reasoning": "step-by-step thought process",
"answer": "final output"
```

Or use `response_mime_type="application/json"` (Gemini) to force structured output without reasoning leaking into the response.

---

## Examples

### Good/bad pairs beat descriptions

Instead of "write natural narrations," show the contrast:

```xml
<good_output>
{"narration": "Someone stood up from the desk."}
</good_output>
<bad_output>
{"narration": "A person is standing in the room, facing forward."}
Why bad: describes static state, not the change.
</bad_output>
```

Cursor uses `<good-example>` and `<bad-example>` tags extensively. Bad examples outnumber good ones 3:1 in Cursor's `<citing_code>` section — teaching what NOT to do is more effective for edge cases.

### Include the trigger/input with examples

Don't just show the output — show what input produced it:
```
Trigger: pose_change sitting->standing, zone_transition desk->door
Output: {"narration": "Someone stood up and walked toward the door."}
```

This teaches the model the mapping from input to output, not just what good output looks like.

### Few-shot quantity guidelines

- Simple tasks: zero-shot can work with strong models
- Complex grammars: multiple examples per command/pattern
- Chain-of-thought: include examples showing reasoning process
- Balance: more examples improve reliability but consume token budget

---

## Identity & role

### One-sentence identity in the opening

Cursor: "You are an AI coding assistant, powered by GPT-4.1."
Devin: "You are Devin, a software engineer using a real computer."

Identity is tied to specific capabilities and scope boundaries, not abstract personality traits. Keep it functional: what the model IS, what it receives, what its output drives.

### Pre-commit responses for identity queries

Devin includes a pre-written response for "who are you?" questions, preventing the model from improvising an answer that might break character. Useful when the model's identity needs to be consistent.

---

## Reliability

### Chain-of-thought for hard decisions

When the model produces wrong answers despite correct setup, add reasoning:
- "Let's think step-by-step" improves accuracy for math/logic
- Show intermediate reasoning in structured format
- Use `thinking_budget` (Gemini) or separate reasoning fields

### Verification checkpoints

Devin embeds "think before acting" gates at critical decision points:
- "Use the think tool before critical git decisions"
- "When transitioning from exploring to making changes"
- "Before reporting completion, critically examine your work"

### Recursive error handling

Prevent infinite retry loops:
- "On the third failure, stop and ask the user" (Devin)
- "If you fail to edit a file, read it again first" (Cursor)
- Set explicit retry limits rather than hoping the model gives up.

### Exponential backoff on API failures

When calling external APIs from a prompt-driven system, build backoff into the architecture, not the prompt. Prompts can't reliably track timing — code handles this better.

---

## Anti-patterns

### Don't rely on prompts as security boundaries

Brex: "Always assume a determined user will bypass constraints." Prompts are behavioral guidance, not access control. Never embed secrets expecting the prompt to protect them.

### Don't over-abstract

Brex: "Avoid too-low (pixel-level) or too-high (single-command) abstraction." Give the model primitives it can compose, not micro-instructions it must follow robotically or mega-instructions it can't decompose.

### Don't use vague guidance when you can show examples

"Write good narrations" is useless. Show 3 good/bad pairs and the model calibrates. Description tells, examples show.



AmbiguityResolve_prompt_hitab = """
You are given a question and the column/row headers of a table. Your task has two parts:

1. **Schema Linking**: Identify the relevant column and row headers needed to answer the question, based on any possible linkage between each keyword and the table schema elements. Note sometimes the single keyword in the question may require multiple headers to answer it. 
2. **Question Rewriting**: Reformulate the ambiguous question into a clear, unambiguous version that explicitly references the identified schema elements.

**Important**: Rank your schema linking results by confidence, listing the most likely match first. Output as a JSON array of objects, each with "relevant_schema" and "rewritten_question" keys.
**Response Format:**
{{"schema_linking_results":[
  {{
    "relevant_schema": [relevant schema elements],
    "rewritten_question": "Clear question referencing [schema element 1]"
  }},
  {{
    "relevant_schema": [alternative relevant schema elements],
    "rewritten_question": "Alternative clear question..."
  }}
]}}

---

## Example

**Question:**
"Over the period 2000 to 2016, how many percentage points of capital goods exports to the United Kingdom have declined?"

**Schema:**
{{
  "column_headers": [
    "2000, billions of dollars",
    "2016, billions of dollars", 
    "share in 2000, %",
    "share in 2016, %",
    "growth rate 2000 to 2016, %"
  ],
  "row_headers": [
    "imports of capital goods, united states",
    "imports of capital goods, united kingdom",
    "exports of capital goods, united states",
    "exports of capital goods, united kingdom"
  ]
}}

**Response:**
{{"schema_linking_results":[
  {{
    "relevant_schema": ["growth rate 2000 to 2016, %", "exports of capital goods, united kingdom"],
    "rewritten_question": "What is the opposite value of [growth rate 2000 to 2016, %] for [exports of capital goods, united kingdom]?"
  }}
]}}

---

## Your Task

**Question:** {question}
**Schema:** {table_metadata}
**Response:**
"""
AmbiguityResolve_prompt_real = '''
You are a table analysis assistant. Given a question and a table's schema (column/row headers), determine if the question is ambiguous when answered using the table.

## Task Definition

1. **Ambiguity Assessment**: First, decide if the question is ambiguous in the context of the provided schema.
   - A question is **ambiguous** if it cannot be directly mapped to specific table elements without interpretation, or if keywords in the question could link to multiple schema elements.

2. **If Ambiguous**:
   a. **Schema Linking**: Identify all plausible links between question keywords and schema elements (columns/rows). A single keyword may require multiple headers.
   b. **Question Rewriting**: For each plausible linkage, rewrite the question into an unambiguous form that explicitly references the specific schema elements.
   c. **Confidence Ranking**: Rank the results from most to least confident.

3. **If Clear/Unambiguous**: Return an empty result array.

## Output Format
Respond with a valid JSON object containing a single key: `"schema_linking_results"`.

- If ambiguous: This should be a non-empty array of objects, each with:
  - `"relevant_schema"`: Array of specific column/row headers needed.
  - `"rewritten_question"`: The clear, rewritten question referencing those headers.
- If clear: Return an empty array: `{{"schema_linking_results": []}}`

## Examples

### Example 1: Ambiguous Question
**Question:** "Over the period 2000 to 2016, how many percentage points of capital goods exports to the United Kingdom have declined?"

**Schema:**
{{
  "column_headers": [
    "2000, billions of dollars",
    "2016, billions of dollars", 
    "share in 2000, %",
    "share in 2016, %",
    "growth rate 2000 to 2016, %"
  ],
  "row_headers": [
    "imports of capital goods, united states",
    "imports of capital goods, united kingdom",
    "exports of capital goods, united states",
    "exports of capital goods, united kingdom"
  ]
}}

**Response:**
{{
  "schema_linking_results": [
    {{
      "relevant_schema": ["growth rate 2000 to 2016, %", "exports of capital goods, united kingdom"],
      "rewritten_question": "What is the opposite value of [growth rate 2000 to 2016, %] for [exports of capital goods, united kingdom]?"
    }}
  ]
}}

### Example 2: Clear Question
**Question:** "What is the growth rate of capital goods exports to the United Kingdom from 2000 to 2016?"

**Schema:**
{{
  "column_headers": ["growth rate 2000 to 2016, %"],
  "row_headers": ["exports of capital goods, united kingdom"]
}}

**Response:**
{{"schema_linking_results": []}}

## Your Task

**Question:** {question}
**Schema:** {table_metadata}

**Response (JSON only):**
'''

QuestionRefine_prompt = '''
# Task
To combine an `original_question` with `additional_information` into a single, coherent, and complete new question that is logically sound and easy to understand.

# Core Principles
1.  **Absolute Preservation**: You MUST preserve ALL constraints, details, and intents from the `original_question`. Nothing from the original should be omitted or altered unless it is directly and explicitly contradicted by the `additional_information`.
2.  **Full Integration**: You MUST seamlessly integrate ALL new requirements and constraints from the `additional_information` into the new question.
3.  **Conflict Resolution**: If a piece of `additional_information` directly conflicts with a part of the `original_question`, the `additional_information` takes precedence and should be used to update or replace the conflicting part. This is the **only** scenario where original information may be modified.
4.  **Natural Language**: The final output must be a single, natural-sounding question, not a list of criteria.

# Examples
Original question: List all novels published after 2000 that won a Booker Prize.
Additional information: Only include novels that were also adapted into movies and written by female authors.
Rewritten question: List all novels published after 2000 that won a Booker Prize, were adapted into movies, and were written by female authors.

Original question: Which Asian countries have a GDP per capita above $30,000 and a population under 10 million?
Additional information: Exclude countries that are island nations.
Rewritten question: Which Asian countries that are not island nations have a GDP per capita above $30,000 and a population under 10 million?

Original question: Provide the list of Olympic gold medalists in swimming events for the last three Summer Olympics, including their ages at the time of winning.
Additional information: I am only interested in male athletes from North America, and only in individual events.
Rewritten question: Provide the list of male North American Olympic gold medalists in individual swimming events for the last three Summer Olympics, including their ages at the time of winning.

# Response Format
- Return **only** the text of the rewritten question.
- Do not include any preamble, labels (like "Rewritten question:"), or explanations.

---

Original question: {question}

Additional information: {additional_info}

Rewritten question:
'''

QuestionRewrite_prompt = '''
Rewrite the given question using the provided relevant_schema so that the question is precise and can be answered using those schema elements, while preserving the original meaning.

Instructions:
- Incorporate all items from relevant_schema explicitly into the rewritten question.
- Do not alter the core intent of the original question.
- Return only the rewritten question text—no labels, no explanations.

Example:
Original question: over the period 2000 to 2016, how many percentage points of capital goods exports to the united kingdom has declined?
Relevant_schema: ["exports of capital goods, united kingdom", "growth rate 2000 to 2016, %"]
Rewritten question: What is the opposite value of [growth rate 2000 to 2016, %] of [exports of capital goods, united kingdom].

---
Input
Original question: {question}
Relevant_schema: {relevant_schema}
Rewritten question: 
'''

RewriteClarificationQuestion_prompt = '''
# Role
You are an expert AI assistant that excels at simplifying complex technical information into clear, user-friendly, multiple-choice options.

# Task
Your task is to analyze a clarification question and its accompanying description. Based on this, you must generate a list of choices. 
Each choice should be a self-contained, natural language sentence that is easy for a non-technical user to understand and select.
- Make sure all choices follow similar formats (e.g, choice + explanation/evidence) 
- If there is a "Unclear column reference", list all column choices with "column_name, table_name, column_description" in a descriptive sentence.

## Input
- **Question**: The clarification question that needs to be answered.
- **Description**: The context or data containing the potential answers. This can be a simple string or a structured JSON object.

## Output format
You MUST respond with ONLY a single, valid JSON object. The object must contain a single key, "choices", which is a list of strings. Do NOT add any other text, explanations, or markdown formatting.

Input question: {question}
Input description: {description}
### Example:
---
**Input:**
- Question: "Do you mean drivers born after the end day or the end year of the Vietnam War?"
- Description: "The end day is 1975-04-30, the end year is 1975."

**Correct Output:**
```json
{{
  "choices": [
    "End Day: April 30, 1975.",
    "End Year: Dec 31, 1975."
  ]
}}
'''


UncertaintyIdentifyAndGround_prompt = """
You are a table-question grounding assistant.

You are given:
(1) a question, and
(2) table metadata node labels (unique labels), consisting of:
    - subtable_titles: section/subtable titles
    - column_headers: column header node labels
    - row_headers: row header node labels

Your task has TWO parts:

A) Identify underspecified (ambiguous) query phrases
- Extract all phrases in the question that are underspecified/ambiguous in the context of the table metadata.
- A phrase is underspecified if it cannot be grounded to a specific table element without interpretation,
  OR it could refer to multiple metadata nodes.
- If there is NO underspecified phrase, return {{"phrase_groundings": []}}.
- Do not extract a vague noun phrase as a separate ambiguity when the question
  already supplies its comparison axis. For example, in "by household size,
  which kind of people...", ground the household-size axis and do not reinterpret
  "kind of people" as age, sex, education, or another competing dimension.

B) Ground each phrase to relevant metadata node labels (one-to-many)
- For EACH underspecified phrase, select ALL relevant node labels it could refer to.
- Return selections in three buckets: subtable_titles / column_headers / row_headers.
- Selections must be STRICTLY chosen from the provided metadata lists (no invented labels).
- Rank the selected labels within each bucket by confidence (most likely first).
- Prefer the narrowest node(s) that satisfy explicit query constraints. Select
  a parent node only when the question genuinely requests all of its children.
- If a bucket has no relevant labels, return an empty list for that bucket.

STRICT RULES
- Use ONLY labels that appear in the provided metadata lists (verbatim; no paraphrasing).
- Do NOT do any global consistency or cross-phrase pruning. (We will handle that later.)
- Output MUST be valid JSON and nothing else.

OUTPUT JSON FORMAT:
{{
  "phrase_groundings": [
    {{
      "phrase": "...",
      "selected_nodes": {{
        "subtable_titles": [...],
        "column_headers": [...],
        "row_headers": [...]
      }}
    }}
  ]
}}

Question: {question}
Metadata: {table_metadata}
"""

UncertaintyRewrite_prompt = """
You are given:
(1) an original question,
(2) a list of key query phrases extracted from the question (key_phrases), and
(3) a set of relevant structural elements (as strings) that can be used to ground those phrases.

Your task:
Internally consider the plausible interpretations, compare them against the
original wording and table hierarchy, and output ONE canonical interpretation.
The canonical result must include a rewritten question, the exact row/column
schema paths needed, explicit scope decisions, and an operator hint.

How to use the inputs:
- For each phrase in key_phrases, select one or more relevant strings from structural_elements that best ground it.
- Use the selected strings verbatim (do not paraphrase or edit them).
- You may insert the selected strings as bracketed qualifiers, but the string itself must remain unchanged.

Rules:
- Preserve the original intent and semantics of the question.
- Preserve every explicit qualifier from the original question, especially
  parenthetical definitions, numbers, years, ranges, units, and comparison scope.
- Never emit the literal strings "None", "null", or "nan" as structural elements.
- Do NOT add new facts beyond what is implied by the selected structural elements.
- Use ONLY strings from structural_elements when grounding (no invented labels).
- If a phrase cannot be grounded by any provided string, leave it unchanged and record an empty selection for it.
- Prefer a specific leaf path over a parent path when the parent has multiple
  descendants. If the question asks for an overall/general value without
  naming a subgroup, prefer an explicit total/all/overall leaf when available.
- If the question asks "for each" or explicitly names several groups, retain
  all of those groups; otherwise choose one canonical answer scope.
- relevant_schema.row_headers and relevant_schema.column_headers must contain
  only verbatim strings present in Table Metadata.
- operator_hint must be one of: direct, sum, average, count, min, max,
  difference, decline_magnitude, difference_rate, proportion, opposite,
  argmin, argmax.
- "how many times" / "times as high" means proportion, not difference_rate.
- "percentage points" means a difference; if it asks how much something
  declined/fell, use decline_magnitude (old value minus new value).
- "percent increase/decrease" means difference_rate.
- "accounted for what percent/share of total" means proportion, possibly
  after summing the named components.

Generate up to THREE mutually distinct candidate interpretations. Candidates
must differ in a real ambiguity (scope, axis, total/subgroup, period, or
operator), not wording. Put the most likely candidate first. Do not merge
incompatible axes into one candidate. An explicit constraint in the question
(for example "by household size") outranks a vague phrase (for example
"kind of people").

Output MUST be valid JSON and nothing else.

OUTPUT JSON FORMAT:
{{"candidates": [
  {{
    "rewritten_question": "...",
    "relevant_schema": {{
      "row_headers": ["exact | row path"],
      "column_headers": ["exact | column path"]
    }},
    "scope_decisions": {{"ambiguous phrase": "chosen scope"}},
    "operator_hint": "proportion"
  }}
]}}

Original Question: {question}
Key Phrases: {key_phrases}
Structural Elements: {structural_elements}
Groundings By Phrase: {grounding_by_phrase}
Table Metadata: {table_metadata}
"""

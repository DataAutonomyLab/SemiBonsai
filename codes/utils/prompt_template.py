evaluation_prompt_en = """
### Instruction
You are given two values, A and B, and both are answers to the same question. A is the correct answer, and B is another person's answer. Determine whether B is correct based on A. Output T if B is correct, and F if it is incorrect.
If A and B are numerically equal but have different units (e.g., 498 vs. 498 billion yuan, 98 vs. 98 people), consider B correct.

### Input
A: {a}
B: {b}

### Note
Output T or F only. T indicates the answer is correct, and F indicates it is incorrect. Do not provide any explanation or additional formatting.
"""



evaluation_prompt_num = """
Compare two numeric answers and output only T or F.

Input
A: {a}
B: {b}

Rules
1) Parse the first number and unit in A,B; if missing → F.
2) Normalize units (same dimension only):
   %→×0.01; 千/万/亿→×1e3/1e4/1e8; thousand/million/billion/trillion→×1e3/1e6/1e9/1e12;
   mm/cm/m/km→×1e-3/1e-2/1/1e3; mg/g/kg→×1e-6/1e-3/1; ms/s/min/h→×1e-3/1/60/3600;
   currency/count labels (¥/$/€/人/people) → ×1. Incompatible dimensions → F.
3) Let nA,nB be normalized. If |nA−nB| ≤ max(1e-6, 0.02*max(|nA|,|nB|)) → T.
4) Else, if nB/nA is within 2% of any of {{0.01, 0.1, 1, 10, 100, 1e3, 1e4, 1e6, 1e8, 1e9, 1e12}} → T.
5) Both ≈ 0 (≤1e-6) → T; otherwise F.
"""


identify_subtable_prompt_multiple = """
You are a table-understanding agent. Given a TABLE IMAGE (with or without OCR/layout), decide whether the page contains exactly ONE subtable or MULTIPLE subtables.

Output format (STRICT):
- Output ONLY a JSON object with ONE key:
  {"single_subtable": "yes"}  OR  {"single_subtable": "no"}
- Do NOT output any other keys.
- Do NOT output any explanation text.

How to decide MULTIPLE subtables ("single_subtable": "no"):
Treat the page as containing multiple subtables if it can be segmented into TWO OR MORE distinct vertical sections, where EACH section satisfies ALL of the following:
1) It forms a separate block/section, typically separated by full-width divider rows or section-title rows spanning the table width.
2) It has its own explicit column-header band (i.e., its own header row(s) for the data columns), not shared with other sections.
3) It has its own explicit row headers / entity rows under its header band (i.e., a distinct set of rows that belong to that section).

Additional guidance:
- If the page shows repeated header bands (e.g., a header row appears again mid-table), this is strong evidence of multiple subtables.
- If there are clear section titles that span across the full width and the table restarts its headers below, this is strong evidence of multiple subtables.
- If the table has only one header band at the top and all rows belong to a single continuous body, treat it as a single subtable.

Now decide and output the JSON.
"""

identify_subtable_prompt = """
You are a table-understanding agent. Given a TABLE IMAGE (with or without OCR/layout), extract the table’s header structure and return a JSON object in the following format:

{
  "column_header_groups": [
    // From the top header band, left→right.
    // Multi-level example: { "group": "Year", "children": ["2022"] }
    // Flat example:        { "group": null,   "children": ["Net Sales"] }
  ],
  "has_row_header": "yes" | "no",  // "yes" only if the left-most column has mostly-unique values AND no merged cells
  "row_header_groups": [
    // Include ONLY when has_row_header = "yes".
    // If hierarchical (indentation/parent rows): { "group": "<parent>", "children": ["<child1>","<child2>", ...] }
    // If flat:                                   { "group": null, "children": ["<row1>","<row2>", ...] }
  ],
  "summary_text": "<'Total'/'Subtotal' if present, else ''>"
}

Column headers
- Identify the leaf headers: header cells aligned 1-to-1 with each data column (exclude left stub/row-header columns), ordered left→right.
- If there is any spanning/merged header cell in the header band (colspan > 1) OR a shared unit/rate label (e.g., "percentage", "%", "rate", "ratio") that visually covers multiple adjacent leaf columns, treat it as a nested (grouping) header.
  - If the spanning/shared header is ABOVE the leaf headers: output
    { "group": "<span text>", "children": ["<leaf1>", "<leaf2>", ...] }.
  - If the spanning/shared header is BELOW the leaf headers: for every covered leaf column, output
    { "group": "<leaf>", "children": ["<span text>"] }.
- If no spanning/shared headers exist, output flat items:
  { "group": null, "children": ["<leaf>"] } for each leaf column.
- Do NOT output both a flat item and spanning-based items for the same columns.
- Use ONLY texts that appear in table cells (trim only; do not invent or normalize labels).

Row headers (entity column)
- Check the left-most column (below the header band).
- Set has_row_header = "yes" only if:
  1) Most non-empty values are unique, AND
  2) Cells are NOT merged (i.e., no vertical merges acting as section titles).
- When has_row_header = "yes", build row_header_groups (preserve hierarchy if visible). Otherwise, leave it empty.

Rules
- Do not mix columns from different visual blocks.
- Do not invent or normalize strings—use verbatim text from the image.
- Output ONLY the JSON object described above (no extra prose).

Examples:
"""



# query_plan_prompt = """You plan queries over a TABLE GRAPH (nodes: row_header leaves, column_header leaves, value cells).
# Return 1–3 executable plans. Each plan contains: a reasoning program with placeholders, retrieval subplans to fill them, and an execution order. Do not invent unseen headers/values.

# AVAILABLE RETRIEVAL OPS (graph-side; build a slice T)
# - ROWS(match: str)                              → list[row_header_node_ids]
# - COLS(match: str)                              → list[column_header_node_ids]
# - ATTACH_VALUES(rows, cols)                     → T  (rows×cols matrix)
# - FILTER_ROWS(contains?: str, regex?: str) on T → T
# - FILTER_COLS(contains?: str, regex?: str) on T → T
# - FILTER_ROWS_EQ(col_contains: str, value: str) → list[row_header_node_ids]
#     Keep only rows whose cell under the first column matching `col_contains`
#     equals `value` (runtime resolves value via exact→fuzzy→embedding).

# - RESOLVE_VALUE(col_contains: str, query: str) → {{"column","column_label"}}
#     Returns the best-matching existing value under that column (for later use).

# - FILTER_BY_NUMERIC(col_contains: str, op: one of [">","<",">=","<=","==","!="], rhs: number) on T → T
#     Keeps rows whose cell in the column matching `col_contains` satisfies the predicate.

# (If a needed value isn’t directly/fuzzily found, runtime may do value-index matching; you just supply the intended substrings.)

# AVAILABLE REASONING OPS (table-side; run on T)
# - SELECT_CELL(row_contains?: str, col_contains?: str) → {{"row","col","value"}}
# - ARGMAX(axis: "table"|"row"|"col")                   → {{"row","col","value","float"}}
# - ARGMIN(axis: "table"|"row"|"col")                   → {{"row","col","value","float"}}
# - AGG(fn: "sum"|"avg"|"min"|"max", axis?: "table"|"row"|"col") → number
# - SORT(by: "row"|"col", numeric?: bool=true, ascending?: bool=true) → T
# - LIMIT(k: int, axis: "row"|"col") → T


# RULES
# 1) First write a REASONING PROGRAM using placeholders (#1, #2, …). Example: "(#1 / #2)".
#    Each placeholder corresponds to one scalar.
# 2) For each placeholder, provide exactly one retrieval subplan (steps to build T) and one reasoning op on T to bind the scalar.
# 3) Deduplicate retrieval inside a plan: reuse the same subplan_id for identical slices.
# 4) Prefer precise substrings copied from the provided headers; keep steps minimal and deterministic.
# 5) Output ONLY the JSON below—no prose.

# INPUTS
# - Question: 
#   {question}
# - Table Metadata:
#   {table_meta}

# OUTPUT JSON SCHEMA
# {{
#   "plans": [
#     {{
#       "plan_id": "P1",
#       "program": "<expression using #1, #2, ...>",
#       "placeholders": [
#         {{"id":"#1","intent":"what #1 represents"}},
#         {{"id":"#2","intent":"what #2 represents"}}
#       ],
#       "retrieve_subplans": [
#         {{
#           "id":"subA",
#           "steps":[
#             {{"op":"ROWS","match":"<substring from row_headers>"}},
#             {{"op":"COLS","match":"<substring from column_headers>"}},
#             {{"op":"ATTACH_VALUES"}},
#             {{"op":"FILTER_ROWS","contains":"<opt>"}},
#             {{"op":"FILTER_COLS","contains":"<opt>"}}
#           ]
#         }}
#       ],
#       "bindings": [
#         {{"id":"#1","retrieve_subplan_id":"subA","reasoning_on_T":{{"op":"SELECT_CELL","row_contains":"...","col_contains":"..."}}}},
#         {{"id":"#2","retrieve_subplan_id":"subB","reasoning_on_T":{{"op":"ARGMAX","axis":"col"}}}}
#       ],
#       "execution": {{
#         "order": ["subA","subB","program"],
#         "can_parallelize": true
#       }}
#     }}
#   ]
# }}
# """

# query_plan_prompt = """
# You are given a QUESTION and TABLE METADATA (row/column headers, optional hints).
# Produce multiple QUERY PLANS if appropriate; otherwise, produce a single plan.
# Each plan is PROGRAM-FIRST: write a final math formula (the program) over placeholders, then add one minimal subquestion per placeholder to fetch data.

# STRICT OUTPUT RULES
# - RETURN **VALID JSON** that can be parsed by Python json.loads.
# - Use only standard JSON (double quotes, no comments, no trailing commas, no markdown fences).
# - Do not include any fields other than those specified below.
# - Every string must be plain text (escape inner quotes if needed).

# RULES
# 1) VERBATIM HEADERS ONLY: copy substrings exactly from row_headers / column_headers. Do not invent headers or values.
# 2) PROGRAM = PURE MATH FORMULA
#    - Operands: placeholders (#1, #2, …) and numeric literals only.
#    - Operators: +, -, *, /, ^  (use ^ for power).
#    - Allowed funcs: sum(), avg(), min(), max(), abs(), sqrt(), log(), exp().
#    - No prose/units in the program. Examples: avg(#2), (#1 - #2) / #3.
# 3) ONE LEAF PER PLACEHOLDER: for each placeholder that appears in the PROGRAM, add exactly one subquestion that FETCHES the needed table data.
# 4) NO HEADERLESS LEAVES: if a step only aggregates/transforms prior subanswers (and needs no new table data), encode it in the PROGRAM—not as a subquestion.
# 5) SELECTOR vs VALUE SPLIT (prevent fused leaves):
#    - If filtering/selection uses a DIFFERENT row/column than the target values, split into TWO leaves:
#        a) SELECTOR leaf that returns a set of headers (columns/rows) satisfying the condition, and
#        b) VALUE leaf that consumes that set to fetch the target values.
#    - Do NOT mention both a filter row (e.g., "Net Sales") and a target row (e.g., "EBIT") in the same subquestion, unless it is a single-cell lookup.
# 6) MINIMAL & DETERMINISTIC SUBQUESTIONS: state filters/units explicitly (e.g., “(in millions)”, “where ‘Total net revenue’ > 0”).
# 7) NON-EMPTY RELEVANT HEADERS: each subquestion MUST list non-empty relevant_column_headers and/or relevant_row_headers that directly support fetching its subanswer (used to build the subtable).

# INPUTS
# - Question:
#   {question}
# - Table Metadata:
#   {table_meta}

# OUTPUT (STRICTLY this schema)
# {{
#   "plans": [
#     {{
#       "plan_id": "P1",
#       "subquestions": [
#         {{
#           "id": "#1",
#           "subquestion": "<selector OR value subquestion to fetch data>",
#           "relevant_column_headers": ["<copy substrings from column_headers>", "..."],
#           "relevant_row_headers": ["<copy substrings from row_headers>", "..."]
#         }}
#       ],
#       "program": "<pure math formula using #IDs only (e.g., 'avg(#2)', '(#1 - #2) / #3')>"
#     }}
#   ]
# }}
# """

# my previous one
# query_plan_prompt = """
# You are given a QUESTION and TABLE METADATA (row/column headers and optional hints).
# Generate up to 3 QUERY PLANS, but only if each plan is **meaningfully distinct**; otherwise, produce a single best plan. Do not generate redundant or near-duplicate plans.

# STRICT OUTPUT REQUIREMENTS
# - Output must be **valid JSON** parsable by Python json.loads (no comments, no markdown, no trailing commas, double quotes only).
# - Include only the specified fields; all string values must be plain text (escape internal quotes as needed).
# - Do not invent any field, value, or header.

# PLAN DIVERSITY & UNIQUENESS
# - At most 3 plans total.
# - Each plan must differ from others in at least one of:
#   (a) program structure (distinct operators, aggregation, or order),
#   (b) header selection (distinct filters or paths),
#   (c) value scope (distinct valid header subsets).
# - Remove duplicates: plans that are equivalent in logic, headers, or formula should be deduplicated (keep one only).

# RULES
# 1. **EXACT HEADER PHRASES ONLY:** When specifying relevant_column_headers or relevant_row_headers, you must use substrings copied exactly (verbatim) from the provided column_headers or row_headers in TABLE METADATA. **Do not create, modify, or infer header names.**
# 2. **PROGRAM = PURE FORMULA:** 
#    - Program must be a pure mathematical expression using only #IDs and numeric literals.
#    - Allowed operators: +, -, *, /, ^  (for exponent).
#    - Allowed functions: sum(), avg(), min(), max(), abs(), sqrt(), log(), exp().
#    - Do not include prose, units, or explanations in the formula (examples: avg(#2), (#1-#2)/#3).
# 3. **ONE LEAF PER PLACEHOLDER:** For every #ID used in the program, there must be one corresponding subquestion that fetches table data.
# 4. **NO HEADERLESS LEAVES:** Aggregation or transformation steps over prior subanswers should be encoded only in the PROGRAM, not as a subquestion.
# 5. **SELECTOR–VALUE SPLIT:** 
#    - If you need to filter by a different row/column than the value to fetch, split into two subquestions: 
#      (a) SELECTOR leaf (returns headers matching condition), 
#      (b) VALUE leaf (consumes the selection and fetches target values).
#    - Do not combine filter and value logic in a single subquestion, unless it is a direct single-cell lookup.
# 6. **EXPLICIT FILTERS & UNITS:** State units, filters, and value types explicitly in subquestions (e.g., “(in millions)”, “where 'Total net revenue' > 0”).
# 7. **NON-EMPTY RELEVANT HEADERS:** Each subquestion must have non-empty relevant_column_headers and/or relevant_row_headers, selected **only from the TABLE METADATA**.

# INPUTS
# - Question:
#   {question}
# - Table Metadata:
#   {table_meta}

# OUTPUT (STRICTLY this schema)
# {{
#   "plans": [
#     {{
#       "plan_id": "P1",
#       "subquestions": [
#         {{
#           "id": "#1",
#           "subquestion": "<selector OR value subquestion to fetch data>",
#           "relevant_column_headers": ["<exact substrings from column_headers>", "..."],
#           "relevant_row_headers": ["<exact substrings from row_headers>", "..."]
#         }}
#       ],
#       "program": "<pure formula using #IDs only, e.g., 'avg(#2)', '(#1-#2)/#3'>"
#     }}
#   ]
# }}
# """

# query_plan_prompt = """
# You are given a QUESTION and TABLE METADATA (row/column headers and optional hints).
# Generate up to 3 QUERY PLANS, but only if each plan is **meaningfully distinct**; otherwise, produce a single best plan. Do not generate redundant or near-duplicate plans.

# Subquestions should **only fetch headers or values**. Any math reasoning (such as ranking, aggregation, or filtering) must be done via a `sub_program` on that subquestion, or in the final `program`. If the answer requires returning a header (such as the year/section where a value is maximum or minimum), you should:
# - Use one subquestion to fetch the relevant values (e.g., values for a specific row across all columns).
# - Use the `sub_program` in that subquestion to perform the ranking operation (restricted to "max(#id)" or "min(#id)" for ranking).
# - Use a second subquestion (a SELECTOR) to return the header(s) (row or column) where the fetched value(s) match the ranked value computed above, by referencing the output of the previous subquestion (e.g., use #1 in your selector logic).

# STRICT OUTPUT REQUIREMENTS
# - Output must be **valid JSON** parsable by Python json.loads (no comments, no markdown, no trailing commas, double quotes only).
# - Include only the specified fields; all string values must be plain text (escape internal quotes as needed).
# - Do not invent any field, value, or header.

# PLAN DIVERSITY & UNIQUENESS
# - At most 3 plans total.
# - Each plan must differ from others in at least one of:
#   (a) program structure (distinct operators, aggregation, or order),
#   (b) header selection (distinct filters or paths),
#   (c) value scope (distinct valid header subsets).
# - Remove duplicates: plans that are equivalent in logic, headers, or formula should be deduplicated (keep one only).

# RULES
# 1. **EXACT HEADER PHRASES ONLY:** When specifying relevant_column_headers or relevant_row_headers, you must use substrings copied exactly (verbatim) from the provided column_headers or row_headers in TABLE METADATA. **Do not create, modify, or infer header names.**
# 2. **PROGRAM = PURE FORMULA:** 
#    - Program must be a pure mathematical expression using only #IDs and numeric literals.
#    - Allowed operators: +, -, *, /, ^ (for exponent).
#    - Allowed functions: sum(), avg(), min(), max(), abs(), sqrt(), log(), exp().
#    - Do not include prose, units, or explanations in the formula (examples: avg(#2), sum(#2), min(#2), (#1-#2)/#3).
# 3. **ONE LEAF PER PLACEHOLDER:** For every #ID used in the program, there must be one corresponding subquestion that **fetches table headers or values (no math in the subquestion itself)**.
# 4. **NO HEADERLESS LEAVES:** Aggregation or transformation steps over prior subanswers should be encoded only in the PROGRAM, not as a subquestion.
# 5. **SELECTOR–VALUE SPLIT:** 
#    - If you need to filter by a different row/column than the value to fetch, split into two subquestions: 
#      (a) VALUE leaf fetches values.
#      (b) SELECTOR leaf references the ranked value computed by a prior sub_program/program result and returns the matching header(s) **or, if the question seeks a normalized entity (such as year, section, or other logical key) from the headers, phrase the subquestion as “Return the year/section where …”, matching the question’s intent**.
#    - Do not combine filter and value logic in a single subquestion, unless it is a direct single-cell lookup.
# 6. **EXPLICIT FILTERS & UNITS:** State units, filters, and value types explicitly in subquestions (e.g., “(in millions)”, “where 'Total net revenue' > 0”).
# 7. **NON-EMPTY RELEVANT HEADERS:** Each subquestion must have non-empty relevant_column_headers and/or relevant_row_headers, selected **only from the TABLE METADATA**.
# 8. **OPTIONAL SUBPROGRAM IN SUBQUESTION (RANKING ONLY):** If a reasoning step requires a ranking step over the fetched values, include an optional "sub_program" with that subquestion. The subquestion is utilized to fetch the values and the sub_program is utilized to reason over the fetched values.
#    - The "sub_program" uses the **same formula format as "program"**, but is **restricted to ranking operations**: use only "max(#id)" or "min(#id)" to compute the ranked numeric result for that subquestion.
#    - The result of a ranking sub_program is a **numeric value** (e.g., the maximum or minimum). If the final answer must be a header, use a SELECTOR subquestion that compares cell values to this numeric result to return the matching headers.

# INPUTS
# - Question:
#   {question}
# - Table Metadata:
#   {table_meta}

# OUTPUT (STRICTLY this schema)
# {{
#   "plans": [
#     {{
#       "plan_id": "P1",
#       "subquestions": [
#         {{
#           "id": "#1",
#           "subquestion": "<selector OR value subquestion to fetch headers/values>",
#           "relevant_column_headers": ["<exact substrings from column_headers>", "..."],
#           "relevant_row_headers": ["<exact substrings from row_headers>", "..."],
#           "sub_program": "<OPTIONAL (RANKING ONLY): 'max(#id)' or 'min(#id)'>"
#         }}
#       ],
#       "program": "<pure formula using #IDs only, e.g., 'avg(#2)', '(#1-#2)/#3'>"
#     }}
#   ]
# }}
# """


# query_plan_prompt-dd = """
# You are given a QUESTION and TABLE METADATA (row/column headers and optional hints).
# Generate up to 3 QUERY PLANS, but only if each plan is meaningfully distinct; otherwise, produce a single best plan. Do not generate redundant or near-duplicate plans.

# SUBQUESTION SCOPE
# - Subquestions may only fetch headers or values from the table. No math or aggregation inside a subquestion.
# - Any reasoning (ranking, aggregation, arithmetic) must be done via a `sub_program` on that subquestion (ranking only) or in the final `program`.

# WHEN THE ANSWER IS A HEADER (e.g., year/section where a value is max/min)
# 1) Use one subquestion to fetch a vector of candidate values (e.g., a row across all columns).
# 2) In that same subquestion, include a ranking `sub_program` restricted to "max(#id)" or "min(#id)" to compute the numeric ranked value.
# 3) Use a second subquestion (a SELECTOR) that returns the header(s) whose value equals the ranked result by referencing the first subquestion’s output (e.g., "#1").

# STRICT OUTPUT REQUIREMENTS
# - Output must be valid JSON parsable by Python json.loads (no comments/markdown/trailing commas; double quotes only).
# - Include only the specified fields; all strings are plain text (escape internal quotes).
# - Do not invent any field, value, or header.

# PLAN DIVERSITY & UNIQUENESS
# - At most 3 plans total.
# - Each plan must differ from others in at least one of:
#   (a) program structure (different operators or aggregation choice),
#   (b) header selection (different filters/paths),
#   (c) value scope (different valid header subsets).
# - Remove duplicates in logic, headers, or formulas (keep one only).

# RULES
# 1) EXACT HEADER PHRASES ONLY:
#    - For relevant_column_headers / relevant_row_headers, copy substrings verbatim from TABLE METADATA’s column_headers / row_headers.
#    - Do not create, modify, or infer header names.

# 2) PROGRAM TYPES (choose exactly one per plan; **selection order enforced**):
#    **Type A — ARITHMETIC CALCULATION (preferred)**
#    - A pure arithmetic expression over #IDs and numeric literals.
#    - Allowed operators: +, -, *, /, ^ (exponent).
#    - Allowed functions: abs(), sqrt(), log(), exp().
#    - Examples: "(#1-#2)/#3", "abs(#1)", "sqrt(#2)".

#    **Type B — AGGREGATION (use only if arithmetic is not suitable)**
#    - One aggregate over a single #ID: sum(#i), avg(#i), min(#i), max(#i), count(#i).
#    - The #ID must correspond to a subquestion that fetched a vector of numeric (or countable) cells.
#    - Examples: "sum(#2)", "avg(#1)", "count(#3)".

#    **Type C — DIRECT RETURN (use only if neither arithmetic nor aggregation is applicable)**
#    - Directly return a prior subanswer: the program is exactly a single placeholder "#i".
#    - Example: "#2".

#    General constraints:
#    - Use only #IDs and numeric literals; no prose or units.
#    - No nested aggregates (e.g., avg(sum(#1))) and no mixing types; pick exactly one type (A, B, or C).

# 3) ONE LEAF PER PLACEHOLDER:
#    - Every #ID in the program must have exactly one subquestion that fetches headers/values (no math inside the subquestion).

# 4) NO HEADERLESS LEAVES:
#    - Aggregations or transformations over prior subanswers must appear only in the PROGRAM (or ranking via sub_program), never inside subquestions.

# 5) SELECTOR–VALUE SPLIT:
#    - If filtering by a different row/column than the values being fetched, split into two subquestions:
#      (a) VALUE leaf fetches values.
#      (b) SELECTOR leaf references a prior result (#k) and returns matching header(s), or phrases the request as “Return the year/section where …” consistent with the question.
#    - Do not combine filter and value logic in one subquestion unless it is a direct single-cell lookup.

# 6) EXPLICIT FILTERS & UNITS:
#    - State units, filters, and value types explicitly in subquestions (e.g., “(in millions)”, “where 'Total net revenue' > 0”).

# 7) NON-EMPTY RELEVANT HEADERS:
#    - Each subquestion must include non-empty relevant_column_headers and/or relevant_row_headers, taken only from TABLE METADATA.

# 8) OPTIONAL SUBPROGRAM IN SUBQUESTION (RANKING ONLY):
#    - If a step requires ranking, include "sub_program" as "max(#id)" or "min(#id)" (no other functions).
#    - Its result is a numeric value; if the final answer is a header, add a SELECTOR subquestion that matches headers to this numeric result.

# PROGRAM SELECTION ORDER
# - First attempt Type A (Arithmetic). If the question cannot be answered by arithmetic over subanswers, attempt Type B (Aggregation). If Aggregation is also not applicable, use Type C (Direct Return "#i").

# INPUTS
# - Question:
#   {question}
# - Table Metadata:
#   {table_meta}

# OUTPUT (STRICTLY this schema)
# {{
#   "plans": [
#     {{
#       "plan_id": "P1",
#       "subquestions": [
#        {{
#           "id": "#1",
#           "subquestion": "<selector OR value subquestion to fetch headers/values>",
#           "relevant_column_headers": ["<exact substrings from column_headers>", "..."],
#           "relevant_row_headers": ["<exact substrings from row_headers>", "..."],
#           "sub_program": "<OPTIONAL RANKING ONLY: 'max(#id)' or 'min(#id)'>"
#         }}
#       ],
#       "program": "<Type A (arithmetic) OR Type B (aggregation) OR Type C ('#i')>"
#     }}
#   ]
# }}
# """

query_plan_prompt = """
You are given a numerical QUESTION and TABLE METADATA (row/column headers and optional hints).

GOAL
Produce a reasoning PROGRAM and a minimal set of subquestions that, together, exactly compute the answer.

TASKS
(1) Infer the PROGRAM first by identifying the QUESTION’s answer type and mapping it to a canonical program template.
(2) Decompose the QUESTION into subquestions **based on that PROGRAM** so each operand (#1, #2, …) is produced by exactly one subquestion.
(3) If a subquestion is multi-hop, break it into earlier subquestions and connect them via depends_on.
(4) Remove redundant/unused subquestions (dedupe + remap references).
(5) For each final subquestion: (5.1) provide exact relevant headers; (5.2) set requires_program=true only if THIS subquestion performs filtering/ranking/aggregation/groupby/calculation, else false.

TOP-DOWN PROCEDURE
Step 1 — ANSWER TYPE → PROGRAM TEMPLATE (PROGRAM FIRST)
- Detect the answer type from the QUESTION and choose a canonical template, then instantiate operands (#i).
  Examples of common mappings (not exhaustive):
  • Difference:                                 "#1 - #2"
  • Ratio / Rate (unitless):                     "#1 / #2"
  • Percentage increase/decrease:                "(#1 - #2) / #2"
- Choose exactly one program type (see PROGRAM TYPES) that combines #IDs into the final numeric answer.
- Operand coverage: every #i in the PROGRAM must be produced by exactly one table related subquestion; no extra subquestions.

Step 2 — DECOMPOSE BY OPERANDS
- Create the minimal chain so each operand is derivable.
- For multi-hop needs, add earlier subquestions and link them via depends_on (placeholders like “year #1” in text).

Step 3 — DEDUPE & NO UNUSED
- Canonicalize subquestions by their text (with placeholders).
- If two are identical, keep one and replace all references (PROGRAM and depends_on).
- Every remaining subquestion must be used directly by the PROGRAM or transitively via dependencies.

Step 4 — ANNOTATE SUBQUESTIONS
- Fill relevant_column_headers / relevant_row_headers with **exact substrings** from TABLE METADATA.
- Fill depends_on with the IDs this subquestion needs.
- Set requires_program=true only if THIS subquestion itself does filtering/ranking/aggregation/groupby/calculation; otherwise false (even if it depends on a subquestion that required a program).

DEPENDENCY PLACEHOLDERS
- When a subquestion depends on earlier results, reference them in text using placeholders (e.g., “the year immediately before #1”).
- Placeholders may refer only to earlier IDs listed in depends_on.

STRICT OUTPUT REQUIREMENTS
- Output must be valid JSON parseable by Python json.loads (no comments/markdown/trailing commas; double quotes only).
- Include only the fields defined below; strings are plain text (escape internal quotes).
- Do not invent/modify header names; use only substrings present in TABLE METADATA.

PLAN DIVERSITY & UNIQUENESS
- Output up to 3 plans only if meaningfully distinct; otherwise a single best plan.
- Distinctness must differ in at least one of: (a) program structure, (b) header selection, (c) value scope.
- Remove logically duplicate plans (same headers + same formula).

RULES
1) EXACT HEADER PHRASES (for any fetch)
   - relevant_column_headers / relevant_row_headers must be copied verbatim from TABLE METADATA’s column_headers / row_headers.
   - At least one of these lists must be non-empty for any subquestion that fetches table data.

2) PROGRAM TYPES (choose exactly one; selection order enforced)
   Type A — ARITHMETIC (preferred): pure arithmetic over #IDs and numeric literals.
     Allowed: +, -, *, /, ^ ; functions: abs(), sqrt(), log(), exp(). Examples: "#1/#2", "#1-#2".
   Type B — AGGREGATION (only if arithmetic not suitable): one aggregate over a single #ID: sum(#i), avg(#i), min(#i), max(#i), count(#i). The #ID must be a vector from a subquestion.
   Type C — DIRECT RETURN (only if neither arithmetic nor aggregation applies): "#i".
   General: use only #IDs and numeric literals; no prose/units; no nested aggregates; do not mix types.

3) SUBQUESTION CLARITY
   - Subquestion text must be clear and unambiguous.

4) ORDER & DEPENDENCIES
   - Assign IDs in creation order (#1, #2, …) top-down.
   - depends_on may reference only earlier IDs (acyclic).

INPUTS
- Question:
  {question}
- Table Metadata:
  {table_meta}

OUTPUT (STRICTLY this schema)
{{
  "plans": [
    {{
      "plan_id": "P1",
      "program": "<Type A arithmetic | Type B aggregation | Type C '#i'>",
      "subquestions": [
       {{
          "id": "#1",
          "subquestion": "<subquestion text>",
          "relevant_column_headers": ["<exact substrings from column_headers>"],
          "relevant_row_headers": ["<exact substrings from row_headers>"],
          "depends_on": ["#k"],
          "requires_program": true or false,
        }}
      ]
    }}
  ]
}}
"""

llm_table_reasoning_prompt = """
You are given a QUESTION and a TABLE. Answer strictly in JSON.

RULE FOR VALUE-ONLY REQUESTS
- If the QUESTION contains the phrase "only output values" (case-insensitive), return ONLY the values as a JSON array in the order requested:
- Do not use thousands separators for values. Example: -2740, not -2,740.
- If a value appears to be numeric, convert it to a number in the JSON output. Example: $3,333 -> 3333
{{
  "answer": [<value1>, <value2>, ...]
}}
- Otherwise, return a normal JSON answer:
{{
  "answer": "<your answer here>"
}}

INPUTS
- QUESTION:
  {question}
- TABLE:
  {table}
"""

program_table_reasoning_prompt = """
You are given a QUESTION and a TABLE (with rows and columns).

ACTION INPUT
Write Python code that:
1) uses pandas to construct a **minimal** DataFrame named df **directly from the provided TABLE** by selecting only the necessary rows/columns and building df via pd.DataFrame from a Python dict,
2) derives the answer to the QUESTION using df,
3) prints **only** the final answer.

CONSTRAINTS
- Build df **explicitly** with pd.DataFrame from the needed TABLE cells (no full-table parsing).
  Example:
    import pandas as pd
    df = pd.DataFrame({{
      "Exact Column Name A": [<cell_a1>, <cell_a2>, ...],
      "Exact Column Name B": [<cell_b1>, <cell_b2>, ...]
    }})
- Do **not** load or parse raw HTML/text (no pd.read_html, no file I/O, no network).
- Include **only** the columns/rows required for the computation (avoid reconstructing the entire table).
- If needed, filter out semantically irrelevant entries (e.g., totals/subtotals/header rows):
    # e.g., df = df[df["Category"] != "Total"]
- Use standard pandas/numpy operations to select, filter, rank, group, and compute.
- Safely normalize numeric text where relevant (commas, %, parentheses, currency/units).
- The **only** stdout must be the final answer (scalar or a JSON-like Python list).

STRICT OUTPUT FORMAT
Return **exactly** this JSON object:
{{
  "python_code": "<your code here>"
}}

INPUTS
- QUESTION:
  {question}
- TABLE:
  {table}
"""

program_pure_reasoning_prompt = """
You are given a QUESTION.

ACTION INPUT
Write Python code that:
1) derives the answer to the QUESTION,
2) prints **only** the final answer.

STRICT OUTPUT FORMAT
Return **exactly** this JSON object:
{{
  "python_code": "<your code here>"
}}

INPUTS
- QUESTION:
  {question}
"""




llm_ques_reasoning_prompt = """
You are given a QUESTION and its EVIDENCE (prior subanswers). Respond ONLY with JSON.

RULE FOR VALUE-ONLY REQUESTS
- If the QUESTION contains the phrase "only output values" (case-insensitive), return ONLY the values as a JSON array in the order requested:
- Do not use thousands separators for values. Example: -2740, not -2,740.
{{
  "answer": [<value1>, <value2>, ...]
}}
- Otherwise, return a normal JSON answer:
{{
  "answer": "<your answer here>"
}}

INPUT
- QUESTION + EVIDENCE:
  {question}
"""

llm_revise_program_prompt = """
You are given:
- A Python expression: {expr}
- An evaluation environment dictionary: {safe_env}
- An error message from eval: "{error_msg}"

Instructions:
- Analyze the error message.
- Revise the expression or environment as needed to fix the error, while preserving the intent of the original code.
- Output only valid JSON in the format below.

Respond ONLY with JSON:
{{
  "updated_expr": "<revised Python expression>",
  "updated_safe_env": "<revised evaluation environment, as a valid Python dict or string>"
}}
"""

llm_revise_python_prompt = """
You are given:
- A Python code: {python_code}
- An error message from eval: "{error_msg}"

Instructions:
- Analyze the error message.
- Revise the python code as needed to fix the error, while preserving the intent of the original code.
- Output only valid JSON in the format below.

Respond ONLY with JSON:
{{
  "updated_code": "<revised Python code>"
}}
"""

python_program_write_prompt = """
You will produce a short, accurate Python code snippet that computes the numerical answer from the inputs below.

QUESTION: {question}
PROGRAM (uses subquestion ids like #1, #2): {program}
SUBQUESTIONS & SUBANSWERS (list of {{id, subquestion, subanswer}}): {subanswers}

Example:
Input --
QUESTION: "What is the average value of the income?"
PROGRAM: "avg(#1)"
SUBQUESTIONS & SUBANSWERS: [{{"id":"#1","subquestion":"Fetch the values of the income.","subanswer":[200,500]}}]
Output --
{{"python_code":"return sum([200, 500]) / 2"}}

Respond ONLY with JSON:
{{
"python_code": "<Python code here>"
}}
"""

e5_debug_prompt = """
You are a strict debugger that must classify the user's mistake when answering a table question.

Possible debug types (choose exactly one):
1) "select irrelevant data" — the code includes values outside the ground-truth relevant set (wrong scope, wrong rows/columns/units).
2) "missing relevant data" — the code omits values that the ground-truth uses (incomplete selection).
3) "relevant data is correct, but meet with reasoning error" — the selected values match ground-truth, but the computation/program logic is applied incorrectly.

INPUTS
- QUESTION:
{question}

- GROUND_TRUTH_ANNOTATION:
  - relevant_values (natural language statements and/or canonical list):
{ground_truth_values}
  - reasoning_program (e.g., add/divide over placeholders):
{ground_truth_program}

- USER_CODE (data selection + computation):
```python
{user_code}

TASK
Compare the set of values that the USER_CODE selects vs. the ground-truth relevant values. Consider scope (year column), units (e.g., dollars vs. percentages), and filters (e.g., "< 100").
Compare the computation the USER_CODE performs vs. the ground-truth reasoning_program.
Decide ONE debug_type using the rules above.
Provide a terse explanation that cites the concrete mismatch.

OUTPUT (return ONLY this JSON):
{{
"debug_type": "<one of: select irrelevant data | missing relevant data | relevant data is correct, but meet with reasoning error>",
"summary": "<one-sentence, concrete explanation>"
}}
"""

our_debug_prompt = """
You are a strict debugger. The user answers a table question by:
(1) decomposing it into subquestions,
(2) executing each subquestion to get subanswers (retrieved values),
(3) taking the UNION of all subanswers as the retrieved set **R**,
(4) applying a program over **R** to produce the final answer.

Your task: using the **ground-truth annotations** (GT values + GT program), classify the error into exactly ONE type and give a one-sentence explanation.

Allowed debug types (choose exactly one):
1) "select irrelevant data"
2) "missing relevant data"
3) "relevant data is correct, but meet with reasoning error"

DEFINITIONS
- **GT_VALUES (G)**: Canonical set of values that SHOULD be retrieved, per ground-truth annotations (respecting year/column/category/unit/filters implied by the question).
- **GT_PROGRAM**: Ground-truth program that correctly combines **G**.
- **USER_RETRIEVED (R)**: UNION of all subanswers from the user’s subquestions (treat each subquestion’s `result` literally; do not compute on them yet).
- **USER_PROGRAM**: The user’s final program (if provided).
- **Q-CONSTRAINTS**: Constraints implied by the QUESTION (e.g., year=2015, filter=<100, units=dollars, category=Firm sales). Use to interpret GT and diagnose mismatches.

INPUTS
- QUESTION:
{question}

- GT_VALUES:
{ground_truth_values}   // canonical list or normalized statements

- GT_PROGRAM:
{ground_truth_program}

- USER_RETRIEVED (R):   // list of subquestion objects with their `result`
{user_retrieved_values}

- USER_PROGRAM (optional):
{user_program}

PROCEDURE
A) Normalize numbers/units in **G** and the values inside **R** (no arithmetic; preserve duplicates as evidence of over-selection).
B) Compare **R** vs **G**:
   - If **R** contains ANY value not in **G** (violates year/column/category/unit/filters) → **"select irrelevant data"**.
   - Else if **R** is missing ANY value that **G** requires → **"missing relevant data"**.
   - Else (**R** matches **G** with intended multiplicities):
       Compare **USER_PROGRAM** to **GT_PROGRAM** semantically (operators, operands, order, denominator choice, aggregation, filters applied at program time).
       If they differ in a way that changes the composition/aggregation → **"relevant data is correct, but meet with reasoning error"**.

TIE-BREAK RULES
1) If BOTH irrelevant and missing are present in **R** → **"select irrelevant data"**.
2) If NO irrelevant items but some required items are missing → **"missing relevant data"**.
3) Only if **R** matches **G** → check program; if incorrect → **"relevant data is correct, but meet with reasoning error"**.

OUTPUT (return ONLY this JSON):
{{
  "debug_type": "<one of: select irrelevant data | missing relevant data | relevant data is correct, but meet with reasoning error>",
  "summary": "<one concise sentence citing the concrete mismatch; reference offending subquestion IDs and/or the wrong program step>"
}}
"""





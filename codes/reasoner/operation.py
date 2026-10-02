

# ---------------- Agent ----------------
action_generation_prompt = """
You are a decision-making agent. 
Your primary goal is to detect the resolved question in the `subquestions` list and select the next **ALLOWED ACTION** to process it.
You MUST reference the `action history` to determine which questions have not yet been fully broken down.

ALLOWED ACTIONS (choose exactly one):
- infer_calculation_formula
- multihop_question_decomposition
- generate_execute_program

**CRITICAL RULES:**
1.  **Target Selection:** The resolved question as rules 1), 2) and 3).
2.  **Function Parameter:** For `infer_calculation_formula` and `multihop_question_decomposition`, the **first parameter MUST be the chosen Target Subquestion ID.**
3.  **New IDs:** If you select `infer_calculation_formula` or `multihop_question_decomposition`, you MUST generate **new subquestion IDs** (e.g., #11, #12, #13) that have not been used previously.

RULES
1) If the target subquestion is directly executable from the table schema:
   → Use generate_execute_program.
   Directly executable means you can answer the question by identifying the needed row header(s) and column header(s) from `table metadata` WITHOUT:
   - deriving a math formula, AND
   - requiring an intermediate hop/selection.

   Example
   subquestions: {{"raw": "What is the sum of Entergy Arkansas in First Quarter in 2015?"}}
   table metadata: {{"row_headers": ["Entergy Arkansas","Entergy Louisiana"],
                    "column_headers": ["2015: First Quarter","2016: First Quarter"]}}
   action history: []
   Function:
     generate_execute_program()
   Explanation: Simple aggregation after retrieving the values from "Entergy Arkansas" and "2015: First Quarter"; no decomposition or complex formula needed.

2) If the **Target Subquestion** requires a **MATH FORMULA** derived from the prefined operator set below:
    **Operator Set:**
      {operator_set}
   → Use **infer_calculation_formula**.
   The formula must:
   - Use ONLY placeholders (#1, #2, #3, …).
   - Be composed using only operators selected from the **Operator Set**.
   - Map each placeholder to exactly ONE scalar-fetch description from the table.
   
   Use the **minimal formula** that correctly answers the question.
   If a single operator suffices, do NOT decompose further.


   {dataset_specific_example}

3) If the **Target Subquestion** is a **multi-hop question**
   → Call multihop_question_decomposition.
   - order lists the execution sequence of smaller single-hop question (each hop requires a reasoning step, at most 3 reasoning steps)
   - `subquestions` defines each new single-hop question; later ones reference earlier ones.

   Example
   subquestions: {{"#1": "What is Entergy Louisiana in First Quarter in the year with higher Entergy Arkansas?"}}
   table metadata: {{"row_headers": ["Entergy Arkansas","Entergy Louisiana"],
                    "column_headers": ["2015: First Quarter","2016: First Quarter"]}}
   action history: []
   Function:
     multihop_question_decomposition(
       "#1", /* Chosen Target ID used here */
       ["#11","#12"],
       {{
         "#11": "Which year has the higher value for Entergy Arkansas in First Quarter?",
         "#12": "What is Entergy Louisiana in First Quarter for the year selected by #11?"
       }}
     )
   Explanation: The target subquestion #1 is multi-hop question. To answer #1, you should first find the year by answering #11, then fetch the requested value to answer #12 conditioned on the result from #11.


INPUTS
subquestions: {subquestions}
table metadata: {table_metadata}
action history: {operation_history}
The next operation must be one of: {possible_actions}

OUTPUT FORMAT (STRICT JSON — no code fences, no extra text):
{{
  "function": "<one of the three function calls above, fully specified if parameters are needed>",
  "explanation": "<one brief sentence explaining why this action is appropriate>"
}}
"""

action_generation_prompt_formula_only = """
You are a decision-making agent. 
Your primary goal is to detect the resolved question in the `subquestions` list and select the next **ALLOWED ACTION** to process it.
You MUST reference the `action history` to determine which questions have not yet been fully broken down.

ALLOWED ACTIONS (choose exactly one):
- infer_calculation_formula
- generate_execute_program

**CRITICAL RULES:**
1.  **Target Selection:** The resolved question as rules 1) and 2).
2.  **Function Parameter:** For `infer_calculation_formula`, the **first parameter MUST be the chosen Target Subquestion ID.**
3.  **New IDs:** If you select `infer_calculation_formula`, you MUST generate **new subquestion IDs** (e.g., #11, #12, #13) that have not been used previously.

RULES
1) If the **Target Subquestion** requires a **COMPLEX FORMULA** (e.g., growth rate, percentage change, complex ratio, weighted average):
   → Use **infer_calculation_formula**.
   - formula must use ONLY placeholders #1,#2,#3,... with + - * / and parentheses
   - operands must map each placeholder to ONE scalar-fetch description

   Example
   subquestions: {{"raw": "What is the growth rate of Entergy Arkansas in First Quarter between 2015 and 2016?"}}
   table metadata: {{"row_headers": ["Entergy Arkansas","Entergy Louisiana"],
                    "column_headers": ["2015: First Quarter","2016: First Quarter"]}}
   action history: []
   Function:
     infer_calculation_formula(
       "raw", /* Chosen Target ID used here */
       "(#2 - #1) / #1",
       {{
         "#1": "Entergy Arkansas in 2015: First Quarter",
         "#2": "Entergy Arkansas in 2016: First Quarter"
       }}
     )
   Explanation: "growth rate" implies (new - old) / old.


2) Already-minimal subquestions without complex formula
   → Call generate_execute_program.
   - This writes and runs Python to select only the necessary rows/columns and return a scalar

   Example
   subquestions: {{"raw": "What is the sum of Entergy Arkansas in First Quarter in 2015?"}}
   table metadata: {{"row_headers": ["Entergy Arkansas","Entergy Louisiana"],
                    "column_headers": ["2015: First Quarter","2016: First Quarter"]}}
   action history: []
   Function:
     generate_execute_program()
   Explanation: Simple aggregation over a single data segment; no complex formula needed.

INPUTS
subquestions: {subquestions}
table metadata: {table_metadata}
action history: {operation_history}
The next operation must be one of: {possible_actions}

OUTPUT FORMAT (STRICT JSON — no code fences, no extra text):
{{
  "function": "<one of the three function calls above, fully specified if parameters are needed>",
  "explanation": "<one brief sentence explaining why this action is appropriate>"
}}
"""

action_generation_prompt_multihop_only = """
You are a decision-making agent. 
Your primary goal is to detect the resolved question in the `subquestions` list and select the next **ALLOWED ACTION** to process it.
You MUST reference the `action history` to determine which questions have not yet been fully broken down.

ALLOWED ACTIONS (choose exactly one):
- multihop_question_decomposition
- generate_execute_program

**CRITICAL RULES:**
1.  **Target Selection:** The resolved question as rules 1) and 2).
2.  **Function Parameter:** For `multihop_question_decomposition`, the **first parameter MUST be the chosen Target Subquestion ID.**
3.  **New IDs:** If you select `multihop_question_decomposition`, you MUST generate **new subquestion IDs** (e.g., #11, #12, #13) that have not been used previously.

RULES
1) If the **Target Subquestion** is a **multi-hop question** (More than 3 reasoning steps)
   → Call multihop_question_decomposition.
   - order lists the execution sequence of smaller single-hop question (At most 3 reasoning steps)
   - `subquestions` defines each new single-hop question; later ones may reference earlier ones.

   Example
   subquestions: {{"#1": "What is Entergy Louisiana in First Quarter in the year with higher Entergy Arkansas?"}}
   table metadata: {{"row_headers": ["Entergy Arkansas","Entergy Louisiana"],
                    "column_headers": ["2015: First Quarter","2016: First Quarter"]}}
   action history: []
   Function:
     multihop_question_decomposition(
       "#1", /* Chosen Target ID used here */
       ["#11","#12"],
       {{
         "#11": "Which year has the higher value for Entergy Arkansas in First Quarter?",
         "#12": "What is Entergy Louisiana in First Quarter for the year selected by #1?"
       }}
     )
   Explanation: First find the year (#11), then fetch the requested value conditioned on it (#12).

2) Already-minimal simple single-hop subquestions
   → Call generate_execute_program.
   - This writes and runs Python to select only the necessary rows/columns and return a scalar

   Example
   subquestions: {{"raw": "What is the sum of Entergy Arkansas in First Quarter in 2015?"}}
   table metadata: {{"row_headers": ["Entergy Arkansas","Entergy Louisiana"],
                    "column_headers": ["2015: First Quarter","2016: First Quarter"]}}
   action history: []
   Function:
     generate_execute_program()
   Explanation: Simple aggregation over a single data segment; no decomposition needed.

INPUTS
subquestions: {subquestions}
table metadata: {table_metadata}
action history: {operation_history}
The next operation must be one of: {possible_actions}

OUTPUT FORMAT (STRICT JSON — no code fences, no extra text):
{{
  "function": "<one of the three function calls above, fully specified if parameters are needed>",
  "explanation": "<one brief sentence explaining why this action is appropriate>"
}}
"""

# schema_linking_prompt = """You are an expert schema linker. Your task is to analyze a list of subquestions and the provided table metadata (row and column headers) to identify the minimal set of relevant headers required to answer each subquestion.
# Your final output must be a single JSON dictionary where keys are the subquestion IDs (e.g., "#1") and values are dictionaries containing the relevant headers.
# **The output MUST strictly follow this JSON format:**
# {{
# "subquestion_ID": {{"relevant_row_headers": ["list of relevant row header strings"],"relevant_column_headers": ["list of relevant column header strings"]}}
# }}

# Example:
# Input:
# subquestions: {{"#1": "What is Entergy Arkansas in First Quarter in 2015?"}}
# table metadata: {{"row_headers": ["Entergy Arkansas","Entergy Louisiana"],
#                     "column_headers": ["2015: First Quarter","2016: First Quarter"]}}
# Output:
# {{
# "#1":{{'relevant_row_headers': ["Entergy Arkansas"],'relevant_column_headers':["2015: First Quarter"]}}
# }}

# Input:
# subquestions: {subquestions}
# table metadata: {table_metadata}
# Output:
# """
schema_linking_prompt = """You are an expert schema linker.
TASK
Given subquestions and table metadata, return the MINIMAL headers needed to answer EACH subquestion, based on semantic relevance and lexical overlap between subquestion keywords and table headers.

Grounding hints may also be provided. They were linked to phrases in the original
question using the table hierarchy. When a hint is relevant to a subquestion,
include every applicable leaf header; do not replace a list of leaf headers with
only its parent category.

OUTPUT FORMAT (STRICT)
Return a SINGLE JSON OBJECT. Keys are subquestion IDs (e.g., "#1"). Each value is an object with EXACTLY these two keys:
- "relevant_row_headers": list[str]
- "relevant_column_headers": list[str]

Do NOT include any other keys at the top level or inside values.
Do NOT wrap the result in any outer object (no "subquestions", "schema_linking", etc.).
Do NOT include explanations.

Example:
Input:
subquestions: {{"#1": "What is Entergy Arkansas in First Quarter in 2015?"}}
table metadata: {{"row_headers": ["Entergy Arkansas","Entergy Louisiana"],
                    "column_headers": ["2015: First Quarter","2016: First Quarter"]}}
Output:
{{
"#1":{{"relevant_row_headers": ["Entergy Arkansas"],"relevant_column_headers":["2015: First Quarter"]}}
}}

NOW PROCESS
Input:
subquestions: {subquestions}
table metadata: {table_metadata}
grounding hints: {grounding_hints}
Output:
"""

program_generation_prompt_old = """You are an expert Python programmer tasked with generating a single, complete, and directly executable Python code snippet to answer a 'raw' question based on a provided table.
**The output in the final ```python ... ``` block MUST define and assign the final result to a variable named final_answer.**
Use the 'Atomic subquestions' and 'Reasoning hints' to structure your logic, but ensure the final Python block is a flat, executable script that solves the 'raw' question. You should aim for the most concise and direct solution. When using pandas (pd) for table operations, initialize the DataFrame directly in the code from the table data for clarity and self-containment.

Here are some examples:

Input:
Table: /*
row_header  2015: First Quarter 2016: First Quarter
Entergy Arkansas 100 120
Entergy Louisiana 90 95
*/
All questions: {{'raw': 'What is the average increasing rate of Entergy Arkansas in First Quarter between 2015 and 2016?', '#1': 'Entergy Arkansas in 2015: First Quarter', '#2': 'Entergy Arkansas in 2016: First Quarter'}}
Atomic subquestions: {{'#1': 'Entergy Arkansas in 2015: First Quarter', '#2': 'Entergy Arkansas in 2016: First Quarter'}}
Reasoning hints: ['To answer raw, you should utilize the formula: (#2 - #1) / #1.']

Output:
Thought:
The raw question requires a simple calculation: (#2 - #1) / #1. Values are directly accessible: #1 is 100, #2 is 120. The final code will perform this direct arithmetic operation.
The relevant row column  headers for these two steps are xx and xx respectivly.
```python
# Values directly from the table:
value_2015 = 100
value_2016 = 120
final_answer = (value_2016 - value_2015) / value_2015
# print(final_answer)
```

Input: 
Table: /*
row_header  2015: First Quarter 2016: First Quarter
Entergy Arkansas 100 120
Entergy Louisiana 90 95
*/
All questions: {{'raw': 'What is Entergy Louisiana in First Quarter in the year with higher Entergy Arkansas?', '#1': 'Which year has the higher value for Entergy Arkansas in First Quarter?', '#2': 'What is Entergy Louisiana in First Quarter for the year selected by #1?'}} 
Atomic subquestions: {{'#1': 'Which year has the higher value for Entergy Arkansas in First Quarter?', '#2': 'What is Entergy Louisiana in First Quarter for the year selected by #1?'}} 
Reasoning hints: ['To answer the complex question raw, you should answer simpler subquestions in this sequence: ['#1', '#2'].']

Output:
Thought: 
The question requires address two steps (#1 and #2): 
1) #1: Find the year with the maximum 'Entergy Arkansas' value. 
2) #2: Find the 'Entergy Louisiana' value for that year. 
The relevant row column  headers for these two steps are xx and xx respectivly.
```python
import pandas as pd

# Step 1: Find the year with the highest Entergy Arkansas
df = pd.DataFrame({{
    'Year': [2015, 2016],
    'Entergy Arkansas': [100, 120],
    'Entergy Louisiana': [90, 95]
}})
year_with_max_arkansas = df.loc[df['Entergy Arkansas'].idxmax(), 'Year']

# Step 2: Find Entergy Louisiana for that year
final_answer = df[df['Year'] == year_with_max_arkansas]['Entergy Louisiana'].iloc[0]
# print(final_answer)
```

Now, generate python code to address this question based on the provided table context and other supplemental information:
Input: 
# Table: /* {table} */ 
All questions: {subquestions} 
Atomic subquestions: {atomic_subquestions} 
# Reasoning hints: {reasoning_history}
Output:
Thought:
"""

# program_generation_prompt = """You are an expert Python programmer tasked with generating a single, complete, and directly executable Python code snippet to answer a 'raw' question based on the provided subquestions and relevant subtable data.
# **The output in the final ```python ... ``` block MUST define and assign the final result to a variable named final_answer.**
# Use the 'Atomic subquestions', 'Relevant subtable for atomic subquestions', and 'Reasoning hints' to structure your logic. The final Python block must be a flat, executable script that solves the 'raw' question by first calculating the necessary intermediate values (answers to atomic subquestions) and then combining them.
# When using pandas (pd) for table operations, initialize the DataFrame directly in the code using the data from the **Relevant subtable** to perform the necessary calculation for that step.

# Here are some examples:

# Input:
# Atomic subquestions: {{'#1': 'Entergy Arkansas in 2015: First Quarter', '#2': 'Entergy Arkansas in 2016: First Quarter'}}
# Relevant subtable for atomic subquestions: {{'#1': 
# '/*
# row_header  2015: First Quarter
# Entergy Arkansas 100
# */', 
# '#2': '
# /*
# row_header  2016: First Quarter
# Entergy Arkansas 120
# */
# '}}
# Other questions for your inference: {{'raw': 'What is the average increasing rate of Entergy Arkansas in First Quarter between 2015 and 2016?'}}
# Reasoning hints: ['To answer raw, you should utilize the formula: (#2 - #1) / #1.']

# Output:
# Thought:
# The process requires two intermediate values (#1 and #2) followed by a calculation.
# 1. Answer to #1 is 100, extracted from its subtable.
# 2. Answer to #2 is 120, extracted from its subtable.
# 3. Apply the formula: (Answer #2 - Answer #1) / Answer #1.
# ```python
# # --- Step 1: Answer Atomic Subquestion #1 ---
# answer_1 = 100

# # --- Step 2: Answer Atomic Subquestion #2 ---
# answer_2 = 120

# # --- Step 3: Compute Final Answer using Reasoning Hints ---
# final_answer = (answer_2 - answer_1) / answer_1
# ```

# Input: 
# Atomic subquestions: {{'#1': 'Which year has the higher value for Entergy Arkansas in First Quarter?', '#2': 'What is Entergy Louisiana in First Quarter for the year selected by #1?'}} 
# Relevant subtable for atomic subquestions: {{'#1': 
# '/*
# row_header  2015: First Quarter 2016: First Quarter
# Entergy Arkansas 100 120
# */', 
# '#2': '
# /*
# row_header  2015: First Quarter 2016: First Quarter
# Entergy Louisiana 90 95
# */
# '}}
# Other questions for your inferrence: {{'raw': 'What is Entergy Louisiana in First Quarter in the year with higher Entergy Arkansas?'}}
# Reasoning hints: ['To answer the complex question raw, you should answer simpler subquestions in this sequence: ['#1', '#2'].']

# Output:
# Thought: 
# The question requires two sequential steps (#1 and #2) using their respective subtables.
# Answer to #1: Find the year with the maximum 'Entergy Arkansas' value.
# Answer to #2: Use the year from #1 to find the corresponding 'Entergy Louisiana' value.
# ```python
# import pandas as pd

# # --- Step 1: Answer Atomic Subquestion #1 (Find the year with max Entergy Arkansas) ---
# # Relevant subtable for #1: Only Entergy Arkansas data is needed for comparison.
# df_q1 = pd.DataFrame({{
#     'Year': [2015, 2016],
#     'Entergy Arkansas': [100, 120]
# }})
# answer_1 = df_q1.loc[df_q1['Entergy Arkansas'].idxmax(), 'Year']

# # --- Step 2: Answer Atomic Subquestion #2 (Find Entergy Louisiana for the selected year) ---
# # Relevant subtable for #2: Only Entergy Louisiana data is needed for lookup.
# df_q2 = pd.DataFrame({{
#     'Year': [2015, 2016],
#     'Entergy Louisiana': [90, 95]
# }})
# # The value 'answer_1' (the year) is used as input for this step.
# answer_2 = df_q2[df_q2['Year'] == answer_1]['Entergy Louisiana'].iloc[0]

# # --- Step 3: Compute Final Answer ---
# # The answer to the raw question is the result of the last subquestion.
# final_answer = answer_2
# ```

# Now, generate python code to address this question based on the provided table context and other supplemental information:
# Input: 
# Atomic subquestions: {atomic_subquestions} 
# Relevant subtable for atomic subquestions: {atomic_subquestions_subdata}
# Other questions for your inferrence: {subquestions}
# Reasoning hints: {reasoning_history}

# Output:
# Thought:
# """
program_generation_prompt = """You are an expert Python programmer tasked with generating a single, complete, and directly executable Python code snippet to answer a 'raw' question requiring a numerical value based on the provided subquestions and relevant subtable data.

**CRITICAL OUTPUT RULES:**
1.  **NO COMMENTS:** The code snippet in the ```python ... ``` block MUST NOT contain any comments (lines starting with # or inline #). The code must be clean, correct, and directly executable.
2.  **FINAL ASSIGNMENT:** The output in the final ```python ... ``` block MUST define and assign the final result to a variable named `final_answer`.

Use all the provided information (Atomic subquestions, Relevant subtable, Reasoning hints, and potential Numerical reasoning program evidence) to structure your logic sequentially: first calculate the necessary intermediate values (answers to atomic subquestions) and then combine them to solve the 'raw' question.

### Data Handling and Pandas Constraints

* **Explicit DataFrame Construction:** Construct a **minimal** DataFrame (named `df`, `df_q1`, etc.) **explicitly** using `pd.DataFrame` from a Python dictionary. Use data **ONLY** from the **Relevant subtable**.
    * **Data Normalization:** Safely transform and normalize non-standard text data (e.g., commas, %, currency symbols) into standard numerical values (`float` or `int`) when defining the dictionary lists.
    * **Format:** `df = pd.DataFrame({{"Exact Column Header": [<value1>, <value2>], ...}})`
    
* **Data Selection:** Include **only** the columns and rows required for the current computation.
* **Mandatory Filtering (Anti-Pattern Avoidance):** You **MUST** include code to filter out all semantically irrelevant entries (e.g., **"Total"**, "Subtotal", grand totals, or summary rows) immediately after DataFrame construction, if such entries are present in the source data and would interfere with aggregation.

Example of Filtering and Calculation:
```python
import pandas as pd
df = pd.DataFrame({{
  "Category": ["State", "Federal", "Total"],
  "Value": [100, 200, 300]
}})
df = df[df["Category"] != "Total"]
final_answer = df["Value"].sum()
```

Now, generate python code to address this question based on the provided table context and other supplemental information:
Input: 
Atomic subquestions: {atomic_subquestions} 
Relevant subtable for atomic subquestions: {atomic_subquestions_subdata}
Other questions for your inferrence: {subquestions}
Reasoning hints: {reasoning_history}
Numerical reasoning program evidence: {numerical_reasoning_context}

Output:
"""

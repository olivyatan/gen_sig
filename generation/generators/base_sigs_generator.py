
class BaseSignalGenerator:
    def __init__(self, prompt: str = None, prompt_instructions_version: str = "default", num_signals_per_item: int = 1):
        self.prompt = prompt
        self.num_signals_per_item = num_signals_per_item  # Store the number of signals to generate
        self.json_quotes_replacer = '"'
        self.prompt_instructions_versions = {}  # Dictionary to store multiple prompt instructions

        # Define multiple prompts
        self.prompt_instructions_versions["default"] = f'''
                       Given the product's details above, extract the product characteristic which is the most convincing to make a purchase of the product. 
                       The characteristic you extract must appear in the product's details. Use only words that actually appear in the product's details. 
                       All the characteristic's words should be part of a single sentence in the product's details.
                       Think carefully why such a characteristic would convince a potential buyer to purchase the product. 
                       The characteristic should be informative, yet short, only three or four words.
                       Generate your answer in json in the format: {{{{{self.json_quotes_replacer}characteristic{self.json_quotes_replacer}: {self.json_quotes_replacer}short characteristic{self.json_quotes_replacer}, 
                                                                           {self.json_quotes_replacer}explanation{self.json_quotes_replacer}: {self.json_quotes_replacer}your reasoning why this characteristic is important{self.json_quotes_replacer}}}}}.              
                       '''

        # Define a dynamic multi-signal prompt
        self.prompt_instructions_versions["multi_signal"] = f'''
        Extract up to {self.num_signals_per_item} distinct and compelling product characteristics that encourage purchase. If fewer meet the criteria, generate only those that qualify and avoid irrelevant signals.

        ### **Before extracting, carefully evaluate each product characteristic step by step using the following rules:** 

        - **Each characteristic must be unique** and distinct from the others (no repetitions, no near-synonyms).  
        - **Each characteristic must appear exactly as written in the product's details** as a continuous sequence of words, n-gram, or permutation.  
        - **STRICT RULE:** Do not use any words or partial phrases from the title in the characteristic. If it appears in the title, discard it completely.  
        - **Prioritize valuable, informative, unique, compelling and clear product features** that strongly encourage purchase (e.g.,"High Quality Stainless Steel", "Premium Satin Paper", "Strong Durability"," Lightweight Design", "Quick Charging Time", "Non-Allergic", "Natural stone").  
        - **Ensure each characteristic provides new information** and avoids redundancy.  
        - **Avoid phrases that are vague, incomplete, open-ended, or missing context** (e.g., "For Multiple", "Long Big").    
        - **Avoid overly technical characteristics overloaded with numbers** (e.g., "150W 21V 65.80"," Handle 40","30.5" Waist", "Height 25.5").
        - **Avoid vague, redundant, or ambiguous characteristics** that lack specific product value (e.g., "Handmade Yes", "For Multiple","Actually Fits").  
        - **Avoid characteristics with negative connotations** or that imply the item is flawed or requires maintenance (e.g., "Needs Cleaning/Polishing").
        - **Rewrite attribute-value pairs into more natural, conversational phrases. Avoid simply restating them as 'Attribute: Value'.
        - **Each characteristic should be informative and compelling, yet short, 2 - 4 words. **  

        ---
        ### **Examples of Good & Bad Characteristics**  
        ✅ **Good Characteristics (Concise, Informative, and Unique)**:  
          - **Extra-Wide Elastic Edges** ✅ *(Describes a unique design feature for a better fit.)*  
          - **Deep Wall Fitted Design** ✅ *(Highlights a functional aspect for mattress compatibility.)*  
          - **100% Breathable Cotton** ✅ *(Emphasizes material quality for comfort.)*  
          - **USDA Organic Certified** ✅ *(Adds credibility and trust to the product.)*  
          - **Premium Printing Technologies** ✅ *(Showcases a high-quality production method.)* 
          - **Strong Durability** ✅ *(Emphasizes long-lasting quality, making the product a reliable investment, incentivizes purchase.)*
          - **Designed For All Seasons** ✅ *(Highlights versatility, increasing its practicality and appeal.)*


        - ❌ **Bad Characteristics → ✅ Better Alternatives**  
        - ❌ **"Actually Fits"** *(Vague, lacks a specific feature.)* → ✅ **"Extra-Wide Elastic Edges"**  
        - ❌ **"Four Sizes And Options"** *(Unclear, not a feature.)* → ✅ **"Includes Bonus Small Bag"** *(Highlights a tangible product benefit.)* 
        - ❌ **"56% cotton"** *(Suggests a lower-quality blend, which may be less appealing. )* → ✅ **"Unique Classic Design"** *(Highlights a timeless and versatile style that appeals to a broad range of buyers.)*
        - ❌ **"Needs Cleaning"** *(Implies a flaw and introduces negativity.)* → ✅ **"Pure silver"** *(Highlights product’s quality positively.)*
        - ❌ **"Handle 40"** *(Too technical and less appealing.)* → ✅ **"Playful Print"** *(Introduces a lively and vibrant design element)*
        - ❌ **"Women’S Blazer"** *(Obvious from title)* → ✅ **"Business occassion"** 
        - ❌ **"30.5" Waist"** *(Too technical)* → ✅ **"Versatile Length"**   

        ---
        ### **Output Format:**  
        Generate your response in **JSON format**, ensuring that each characteristic is returned separately with an explanation.

        {{{{  
            {self.json_quotes_replacer}characteristic_1{self.json_quotes_replacer}: {self.json_quotes_replacer}first short characteristic{self.json_quotes_replacer}, 
            {self.json_quotes_replacer}explanation_1{self.json_quotes_replacer}: {self.json_quotes_replacer}Very short explanation of why this characteristic is important and where it appears in the product details{self.json_quotes_replacer},
            ''' + ",\n    ".join([
            f"{self.json_quotes_replacer}characteristic_{i}{self.json_quotes_replacer}: {self.json_quotes_replacer}{i} short characteristic{self.json_quotes_replacer},\n"
            f"{self.json_quotes_replacer}explanation_{i}{self.json_quotes_replacer}: {self.json_quotes_replacer}Very short explanation of why this characteristic is important and where it appears in the product details{self.json_quotes_replacer}"
            for i in range(2, self.num_signals_per_item + 1)
        ]) + '''
        }}}}  
        '''

        if self.num_signals_per_item == 1:
            self.prompt_instructions_versions["improved_rules"] = f'''
Given the product's details above, extract the **most compelling product characteristic** that would convince a buyer to make a purchase.

### **Rules for Extraction:**  
- The characteristic **must** appear in the product's details (as a **continuous phrase, n-gram, or permutation**).  
- STRICT RULE: Do not use any words or partial phrases from the title in the characteristic. If it appears in the title, discard it completely. A characteristic should introduce **new information** not obvious from the title.    
- **Prioritize valuable, informative, unique and clear product features that encourage purchase.** (e.g., "Extra-Wide Elastic Edges" instead of "Actually Fits").  
- Avoid phrases that are cut off, open-ended, missing context(e.g., "Ample Power For Multiple", "For Multiple", "Long Big")
- **Avoid overly technical phrases overloaded with numbers (e.g., 'Model XZ-500435 4.3GHz','150W 21V 65.80'). However, keep meaningful measurements units and numerical values exactly(e.g., "100%", "4mm", "2-Pack", "50cm").  
- **Avoid vague, incomplete, or ambiguous characteristics that lack specific product value** (e.g., "Handmade Yes", "For Multiple"). 
- **Do NOT include product category words (e.g., Leash, Towel, Fabric) if they are already in the title. Keep only the key descriptive characteristic (e.g., "Very Durable" instead of "Very Durable Leash").**
- The characteristic must come from a **single sentence** in the product details.  
- **Keep it concise and short!** Extract **only 3-4 words**—longer characteristics are discouraged.  


### **Examples of Good & Bad Characteristics**  
✅ **Good Characteristics (Concise, Informative, and Unique)**:  
  - **Extra-Wide Elastic Edges** ✅ *(Clearly describes a unique design feature that ensures a better fit.)*  
  - **Deep Wall Fitted Design** ✅ *(Highlights a functional design element for various mattress sizes.)*  
  - **100% Breathable Cotton** ✅ *(Emphasizes material quality and comfort, an important purchasing factor.)*  
  - **USDA Organic Certified** ✅ *(Highlights a unique, credible certification that adds trust and quality.)*  
  - **Premium Printing Technologies** ✅ *(Emphasizes a distinct and high-quality production method, setting the product apart.)*  

❌ **Bad Characteristics (Vague, Meaningless, Redundant, Incomplete, and Unclear)**:  
  - **Actually Fits** ❌ *(Vague, does not describe a tangible feature.)*  
  - **Four Sizes And Options** ❌ *(Incomplete and unclear. "Bonus Small Bag" is a more unique and incentivizing feature.)*   
 - **Luxuriously Soft Towel** ❌ *(“Towel” is redundant as it appears in the title and since it is the product itself. "Luxuriously Soft" is a better, more concise characteristic.)*
 - **Genuine OEM Replacement** ❌ *(Repeats title words, A better choice would be "Genuine")*  


### **Output Format:**  
Generate your response in JSON format:

{{{{
    {self.json_quotes_replacer}characteristic{self.json_quotes_replacer}: {self.json_quotes_replacer}short characteristic{self.json_quotes_replacer}, 
    {self.json_quotes_replacer}explanation{self.json_quotes_replacer}: {self.json_quotes_replacer}reason why this characteristic is important{self.json_quotes_replacer}
}}}}
'''
            # Assign the prompt based on the version- If prompt_instructions_version is not found, it falls back to the default version.
        if self.prompt is None:
            prompt_instructions = self.prompt_instructions_versions.get(
                prompt_instructions_version,
                self.prompt_instructions_versions["default"]
            )

            # Prepend the contextual introduction to the selected instructions
            self.prompt = (
                    ''' 
                    You are an expert salesperson that always excels in providing the best buying tips to buyers given a product of interest they wish to buy. 
    
                    Product details:
                    * Product title: {}
                    * Product features: {}
                    * Product description: {}
                    '''
                    + prompt_instructions
            )

class BaseLabelGenerator_short:
    def __init__(self, prompt: str = None):
        self.prompt = prompt
        self.json_quotes_replacer = '"'
        if self.prompt is None:
            prompt_instructions = f'''
                Based on the product's details and signal provided, assign a label:
                0: Bad Signal
                1: Good Signal
                2: Very Good Signal (Urgent)

                Your response must be in the following JSON format:
                {{{{{self.json_quotes_replacer}signal_label{self.json_quotes_replacer}: <label_number>, 
                   {self.json_quotes_replacer}explanation{self.json_quotes_replacer}: {self.json_quotes_replacer}reason for the label{self.json_quotes_replacer}}}}}
            '''
            self.prompt = '''
                Product details:
                * Title: {}
                * Aspects: {}
                * Description: {}
                * Signal: {}
            ''' + prompt_instructions


class BaseLabelGenerator:
    def __init__(self, prompt: str = None):
        self.prompt = prompt
        self.json_quotes_replacer = '"'
        if self.prompt is None:
            prompt_instructions = f'''
                Based on the product's details and the provided signal, assign a label indicating whether the signal is good. Provide a single reason and a detailed explanation for your decision.

               - Labels: 
               0: Bad Signal. Signal is irrelevant, unclear, or dominated by extraneous elements (e.g., excessive numbers or technical jargon).
               Possible Reasons for Label 0:
                Use the following reasons to justify why the signal is bad. You must include a reason for your decision. Choose one reason from the list below:
                -Repetitive Tokens: Contains redundant words or phrases.
                -Not Clear: Vague or lacks meaningful information.
                -Describes the Store or seller.
                -Too Technical: Focuses only on specs without linking to user benefits.
                -Incorrect Grammar.
                -Excessive Length: Overly verbose, detracting from clarity.
                -Off-Topic: Does not provide relevant or valuable product information.
                -Problematic Numerical Pattern: Contains numerical patterns that are misleading, confusing, or not contextualized properly (e.g., random numbers or irrelevant measurements. 
                -Numerical Dominant: Overloaded with numbers, making it difficult to extract meaningful product information.
                -Too Similar to Title: Signal appears partially in title. 

                *The reason should clearly explain why the signal does not meet the criteria for a good or urgent signal.

                1: Good Signal. The signal is clear, relevant, and provides useful information about the product.
                2: Very Good Signal. Signal incentivizes the user to purchase by highlighting unique, desirable, or urgent features. 


               - Examples: 
                Label 0 (Bad Signal):The signal is irrelevant, unclear, or exhibits numerical dominance. examples: 
                  - 45In. X 3.6 Yds
                  - 40 Pieces Per Pack
                  - Four Sizes And Options
                  - Compatible Part Number
                  - USB Or Battery Powered
                  - 2-Pc. 5-In. Gnomes
                  - Genuine Samsung Da29-00003G
                  - 18V Lithium 2Ah Battery
                  - Fits 26 - 50 Lbs
                  - 50 X 60 In
                  - 2100 # 122444 5 Lumens

                  Label 1 (Good Signal): The signal is clear, relevant, and provides useful information about the product.
                  Examples:
                  - 97% Natural Ingredients
                  - Pristine Condition
                  - Smooth Rotating Design
                  - Durable Stainless Steel
                  - Multi-Sizing Options
                  - USDA Organic Certified
                  - Non-Stick Surface
                  - Strong Wind Speed 

                  Label 2 (Very Good Signal):** The signal incentivizes the user to purchase.
                  Examples:
                  - Natural Solid Wood Material
                  - Great Condition
                  - 5 Years Manufacturer Warranty
                  - Solar Powered


                Your response must be in the following JSON format:
                {{{{{self.json_quotes_replacer}signal_label{self.json_quotes_replacer}: <label_number>, 
                   {self.json_quotes_replacer}reason{self.json_quotes_replacer}: add reason from list for bad signal only, 
                   {self.json_quotes_replacer}explanation{self.json_quotes_replacer}: {self.json_quotes_replacer} detailed explanation for the label of good or bad signal{self.json_quotes_replacer}}}}}
            '''
            self.prompt = '''
                Product details:
                * Title: {}
                * Aspects: {}
                * Description: {}
                * Signal: {}
            ''' + prompt_instructions
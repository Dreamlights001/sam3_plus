"""Prompt Engineering for Anomaly Detection"""


class AnomalyPromptGenerator:
    """Generate effective text prompts for anomaly detection"""
    
    def __init__(self):
        # Base anomaly prompts
        self.base_prompts = [
            "anomaly",
            "defect",
            "damage",
            "irregularity",
            "abnormality",
            "fault",
            "blemish",
            "imperfection",
            "discrepancy",
            "distortion"
        ]
        
        # Context-specific prompts for different categories
        self.category_prompts = {
            "mvtec": {
                "carpet": ["stain", "tear", "hole", "wrinkle"],
                "grid": ["broken line", "missing line", "misalignment"],
                "leather": ["scratch", "crack", "stain", "hole"],
                "tile": ["crack", "chip", "stain", "missing piece"],
                "wood": ["scratch", "dent", "stain", "crack"],
                "bottle": ["broken", "crack", "chip", "contamination"],
                "cable": ["cut", "damage", "insulation defect", "twist"],
                "capsule": ["crack", "missing part", "deformation", "contamination"],
                "hazelnut": ["crack", "hole", "missing part", "discoloration"],
                "metal_nut": ["crack", "missing part", "deformation", "thread defect"],
                "pill": ["crack", "missing part", "deformation", "discoloration"],
                "screw": ["missing head", "thread defect", "bend", "crack"],
                "toothbrush": ["bent bristles", "missing bristles", "damage"],
                "transistor": ["missing lead", "damage", "contamination"],
                "zipper": ["broken tooth", "misalignment", "damage"]
            },
            "visa": {
                "candle": ["broken", "melted", "deformed"],
                "capsules": ["broken", "deformed", "empty"],
                "cashew": ["broken", "discolored", "deformed"],
                "chewinggum": ["broken", "deformed", "contaminated"],
                "fryum": ["broken", "deformed", "contaminated"],
                "macaroni1": ["broken", "deformed", "contaminated"],
                "macaroni2": ["broken", "deformed", "contaminated"],
                "pcb1": ["missing component", "short circuit", "damage"],
                "pcb2": ["missing component", "short circuit", "damage"],
                "pcb3": ["missing component", "short circuit", "damage"],
                "pcb4": ["missing component", "short circuit", "damage"],
                "pipe_fryum": ["broken", "deformed", "contaminated"]
            }
        }
    
    def get_base_prompts(self, num_prompts=5):
        """Get base anomaly prompts"""
        return self.base_prompts[:num_prompts]
    
    def get_category_specific_prompts(self, dataset_name, category, num_prompts=3):
        """Get category-specific anomaly prompts"""
        if dataset_name not in self.category_prompts:
            return []
        if category not in self.category_prompts[dataset_name]:
            return []
        return self.category_prompts[dataset_name][category][:num_prompts]
    
    def generate_mixed_prompts(self, dataset_name, category, base_weight=0.5, category_weight=0.5, num_prompts=8):
        """Generate mixed prompts combining base and category-specific prompts"""
        base_prompts = self.get_base_prompts(int(num_prompts * base_weight))
        category_prompts = self.get_category_specific_prompts(dataset_name, category, int(num_prompts * category_weight))
        
        # Combine and deduplicate
        mixed_prompts = list(set(base_prompts + category_prompts))
        
        # If not enough prompts, add more base prompts
        while len(mixed_prompts) < num_prompts and len(base_prompts) < len(self.base_prompts):
            remaining_base = [p for p in self.base_prompts if p not in mixed_prompts]
            if not remaining_base:
                break
            mixed_prompts.append(remaining_base[0])
        
        return mixed_prompts[:num_prompts]
    
    def generate_contextual_prompts(self, dataset_name, category, include_context=True):
        """Generate contextual prompts with image context"""
        mixed_prompts = self.generate_mixed_prompts(dataset_name, category)
        
        if include_context:
            # Add contextual information to prompts
            contextual_prompts = []
            for prompt in mixed_prompts:
                contextual_prompts.extend([
                    f"{prompt} in {category}",
                    f"{prompt} on {category}",
                    f"{category} with {prompt}"
                ])
            return contextual_prompts
        else:
            return mixed_prompts
    
    def get_prompts_for_inference(self, dataset_name, category, prompt_strategy="mixed", num_prompts=8):
        """Get prompts based on strategy"""
        if prompt_strategy == "base":
            return self.get_base_prompts(num_prompts)
        elif prompt_strategy == "category":
            return self.get_category_specific_prompts(dataset_name, category, num_prompts)
        elif prompt_strategy == "mixed":
            return self.generate_mixed_prompts(dataset_name, category, num_prompts=num_prompts)
        elif prompt_strategy == "contextual":
            return self.generate_contextual_prompts(dataset_name, category)
        else:
            return self.get_base_prompts(num_prompts)

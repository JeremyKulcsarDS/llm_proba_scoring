class PromptBuilder:
    """
    Utility class for building prompts for prompt checking.
    """
    def __init__(self):
        self.prompt = ""

    def add_line(self, prefix, content, sufix):
        if content is not None:
            self.prompt += f"{prefix}{content}{sufix}"
        return self
        
    def build(self):
        return self.prompt
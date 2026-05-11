class ContextBuffer:
    def __init__(self, max_turns=10): self.messages=[]; self.max_turns=max_turns
    def add(self, role, content):
        self.messages.append({"role":role,"content":content})
        if len(self.messages)>self.max_turns*2: self.messages=self.messages[-self.max_turns*2:]
    def get(self): return self.messages.copy()
    def clear(self): self.messages=[]

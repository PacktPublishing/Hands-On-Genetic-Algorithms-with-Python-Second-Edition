from copy import deepcopy


class Applicant():

    def __init__(self, dataset_row):
        self.row = dataset_row.copy()
    
    def get_values(self):
        return [self.get_amount(), self.get_duration(), self.get_checking(), self.get_savings()]
    
    def with_values(self, values):
        return deepcopy(self).set_amount(values[0]).set_duration(values[1]).set_checking(values[2]).set_savings(values[3])
    
    def set_amount(self, new_amount):
        self.row["amount"] = int(new_amount)
        return self
    
    def set_duration(self, new_duration):
        self.row["duration"] = int(new_duration)
        return self
    
    def set_checking(self, new_checking):
        if not (0 <= int(new_checking) <= 3):
            raise AssertionError(f"new_checking value {new_checking} not between 0 and 3")
        
        self.row["checking"] = int(new_checking)
        return self
    
    def set_savings(self, new_savings):
        if not (0 <= int(new_savings) <= 4):
            raise AssertionError(f"new_savings value {new_savings} not between 0 and 4")
        
        self.row["savings"] = int(new_savings)
        return self
    
    def get_amount(self):
        return int(self.row['amount'].iloc[0]) 

    def get_duration(self):
        return int(self.row['duration'].iloc[0]) 
    
    def get_checking(self):
        return int(self.row['checking'].iloc[0]) 
    
    def get_savings(self):
        return int(self.row['savings'].iloc[0]) 
    
    def pretty_print(self):
        print(f"Applicant {self.row.index[0]}: {self.row.T}")

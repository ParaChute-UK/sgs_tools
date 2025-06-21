from dataclasses import dataclass

import pint  # type: ignore


@dataclass(frozen=True)
class simulation_flag:
    res: pint.Quantity
    level: str
    Qh: int
    wind: int
    dynamic_flag: str
    scale: int
    Pr_flag: bool
    regularization_type: str

    def shear(self):
        return f"Qh{self.Qh}W{self.wind}"

    def resolution_mag(self, units="m"):
        return self.res.to(units).magnitude

    def resolution_str(self, units="m"):
        return f"{self.res.to(units):~}".replace(" ", "")

    def Pr_str(self):
        if self.Pr_flag:
            return "Pr"
        else:
            return ""

    def dyn_str(self):
        if self.dynamic_flag:
            return "dyn"
        else:
            return "Smag"

    def __str__(self):
        string = self.resolution_str() + self.level + self.shear() + self.dyn_str()
        if self.dynamic_flag:
            string += f"{self.scale}sc" + self.Pr_str() + self.regularization_type
        return string

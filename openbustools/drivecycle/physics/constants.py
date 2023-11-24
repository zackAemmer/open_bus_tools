class AmbientConditions():
    def __init__(self) -> None:
        self.air_density = 1.293 # kg/m^3
        self.gravity = 9.8 # m/s^2

class MoonConditions():
    def __init__(self) -> None:
        self.air_density = 1.5e-15 # kg/m^3
        self.gravity = 1.62 # m/s^2
class TransitBus():
    def __init__(self) -> None:
        self.mass = 14535 # kg
        self.frontal_area = 8.68 # m^2
        self.drag_coeff = .65
        # self.fixed_roll_res = .006
        # self.variable_roll_res = 4.5e-7 # s^2/m^2
        self.tire_pressure = 3 # bar
        self.wheel_radius = 0.5 # m
        self.trans_eff = .95
        self.trans_gear_ratio = 5.67
        self.converter_eff = .97
        self.aux_load = 5000 # W
        self.motor_eff = .85 # From engine chart
        self.regen_eff = .40 # Amount returned via reg braking
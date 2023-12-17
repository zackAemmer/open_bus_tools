import numpy as np


class EnergyModel():
    """
    A class representing an energy model for a bus.

    Attributes:
        bus (Bus): The bus object.
        conditions (Conditions): The environmental conditions object.

    Methods:
        getEnergyDataFrame(cycle): Returns a DataFrame with energy-related calculations for the given cycle.
        printSummary(cycle): Prints a summary of the energy calculations for the given cycle.
        calcIntensity(cycle, combine=True): Calculates the energy intensity for the given cycle.
        calcTotalEnergy(cycle, combine=True): Calculates the total energy consumption for the given cycle.
        calcTotalPower(cycle, combine=True): Calculates the total power consumption for the given cycle.
        calcTotalLoad(cycle, combine=True): Calculates the total load on the bus for the given cycle.
        calcAeroLoad(cycle): Calculates the aerodynamic load on the bus for the given cycle.
        calcGravLoad(cycle): Calculates the gravitational load on the bus for the given cycle.
        calcRollLoad(cycle): Calculates the rolling resistance load on the bus for the given cycle.
        calcAccelLoad(cycle): Calculates the acceleration load on the bus for the given cycle.
    """
    def __init__(self, bus, conditions) -> None:
        self.bus = bus
        self.conditions = conditions

    def getEnergyDataFrame(self, cycle):
        cycle_df = cycle.to_df()
        cycle_df['F_aero'], cycle_df['F_grav'], cycle_df['F_roll'], cycle_df['F_acc'], cycle_df['F_tot'] = self.calcTotalLoad(cycle, combine=False)
        cycle_df['P_motor'], cycle_df['P_regen'], cycle_df['P_aux'], cycle_df['P_tot'] = self.calcTotalPower(cycle, combine=False)
        return cycle_df

    def printSummary(self, cycle):
        print(f"{self.calcTotalEnergy(cycle):.2f} kWh")
        print(f"{np.sum(cycle.distance)/1000/1.6:.2f} mi")
        print(f"{self.calcIntensity(cycle)*1.6:.2f} Avg. kWh/mi")

    def calcIntensity(self, cycle, combine=True):
        total_energy = self.calcTotalEnergy(cycle)
        intensity = total_energy / (np.sum(cycle.distance) / 1000)
        if combine:
            return np.mean(intensity)
        else:
            return intensity

    def calcTotalEnergy(self, cycle, combine=True):
        total_power = self.calcTotalPower(cycle)
        energy = (total_power * cycle.time) / 60 / 60 / 1000
        if combine:
            return np.sum(energy)
        else:
            return energy

    def calcTotalPower(self, cycle, combine=True):
        f_prop = self.calcTotalLoad(cycle)
        wheel_torque = f_prop * self.bus.wheel_radius
        motor_torque = wheel_torque / (self.bus.trans_gear_ratio * self.bus.trans_eff)
        motor_speed = (cycle.velocity * self.bus.trans_gear_ratio) / self.bus.wheel_radius
        aux_power = self.bus.aux_load / self.bus.converter_eff
        motor_power = (motor_speed * motor_torque) / (self.bus.motor_eff * self.bus.converter_eff)
        regen_power = (motor_speed * motor_torque * self.bus.regen_eff) * (self.bus.motor_eff * self.bus.converter_eff)
        # Applies regen power where f_prop is negative, motor power otherwise
        is_regen = f_prop < 0
        regen_power = is_regen * regen_power
        motor_power = np.invert(is_regen) * motor_power
        total_power = motor_power + regen_power + aux_power
        if combine:
            return total_power
        else:
            return (motor_power, regen_power, aux_power, total_power)

    def calcTotalLoad(self, cycle, combine=True):
        force_estimate = self.calcAeroLoad(cycle) + self.calcGravLoad(cycle) + self.calcRollLoad(cycle) + self.calcAccelLoad(cycle)
        if combine:
            return force_estimate
        else:
            return (self.calcAeroLoad(cycle), self.calcGravLoad(cycle), self.calcRollLoad(cycle), self.calcAccelLoad(cycle), force_estimate)

    def calcAeroLoad(self, cycle):
        f_aero = 0.5 * self.conditions.air_density * np.square(cycle.velocity) * self.bus.frontal_area * self.bus.drag_coeff
        return f_aero

    def calcGravLoad(self, cycle):
        f_grav = self.bus.mass * self.conditions.gravity * np.sin(cycle.theta * np.pi / 180)
        return f_grav

    def calcRollLoad(self, cycle):
        f_roll = self.bus.mass * self.conditions.gravity * (self.bus.fixed_roll_res + self.bus.variable_roll_res * np.square(cycle.velocity))
        # c = .005 + (1 / self.bus.tire_pressure) * (.01 + .0095 * np.square(cycle.velocity*60*60/1000/100))
        # f_roll = self.bus.mass * self.conditions.gravity * c
        return f_roll

    def calcAccelLoad(self, cycle):
        f_accel = self.bus.mass * cycle.acceleration
        return f_accel
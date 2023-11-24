import numpy as np


class DriveCycleEnergyModel():
    def __init__(self, bus, conditions) -> None:
        self.bus = bus
        self.conditions = conditions
    def calcTotalPower(self, traj, combine=True):
        f_prop = self.calcTotalLoad(traj)
        wheel_torque = f_prop * self.bus.wheel_radius
        motor_torque = wheel_torque / (self.bus.trans_gear_ratio * self.bus.trans_eff)
        motor_speed = (traj.vel * self.bus.trans_gear_ratio) / self.bus.wheel_radius
        aux_power = self.bus.aux_load / self.bus.converter_eff
        motor_power = (motor_speed * motor_torque) / (self.bus.motor_eff * self.bus.converter_eff)
        regen_power = (motor_speed * motor_torque * self.bus.regen_eff) * (self.bus.motor_eff * self.bus.converter_eff)
        is_regen = f_prop < 0
        regen_power = is_regen * regen_power
        motor_power = np.invert(is_regen) * motor_power
        total_power = motor_power + regen_power + aux_power
        if combine:
            return total_power
        else:
            return (motor_power, regen_power, aux_power, total_power)
    def calcTotalLoad(self, traj, combine=True):
        energy_estimate = self.calcAeroLoad(traj) + self.calcGravLoad(traj) + self.calcRollLoad(traj) + self.calcAccLoad(traj)
        if combine:
            return energy_estimate
        else:
            return (self.calcAeroLoad(traj), self.calcGravLoad(traj), self.calcRollLoad(traj), self.calcAccLoad(traj), energy_estimate)
    def calcAeroLoad(self, traj):
        f_aero = (0.5 * self.conditions.air_density * np.square(traj.vel)) * self.bus.frontal_area * self.bus.drag_coeff
        return f_aero
    def calcGravLoad(self, traj):
        f_grav = self.bus.mass * self.conditions.gravity * np.sin(traj.theta)
        return f_grav
    def calcRollLoad(self, traj):
        f_roll = (self.bus.fixed_roll_res + self.bus.variable_roll_res * np.square(traj.vel)) * self.bus.mass * self.conditions.gravity
        return f_roll
    def calcAccLoad(self, traj):
        f_acc = self.bus.mass * traj.acc
        return f_acc
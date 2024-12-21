import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Giriş ve çıkış değişkenlerini tanımlama
service_speed = ctrl.Antecedent(np.arange(0, 11, 1), 'service_speed')
food_quality = ctrl.Antecedent(np.arange(0, 11, 1), 'food_quality')
customer_satisfaction = ctrl.Consequent(np.arange(0, 11, 1), 'customer_satisfaction')

# Üyelik fonksiyonlarını tanımlama
service_speed['slow'] = fuzz.trimf(service_speed.universe, [0, 0, 5])
service_speed['normal'] = fuzz.trimf(service_speed.universe, [2, 5, 8])
service_speed['fast'] = fuzz.trimf(service_speed.universe, [5, 10, 10])

food_quality['bad'] = fuzz.trimf(food_quality.universe, [0, 0, 5])
food_quality['good'] = fuzz.trimf(food_quality.universe, [2, 5, 8])
food_quality['excellent'] = fuzz.trimf(food_quality.universe, [5, 10, 10])

customer_satisfaction['low'] = fuzz.trimf(customer_satisfaction.universe, [0, 0, 5])
customer_satisfaction['medium'] = fuzz.trimf(customer_satisfaction.universe, [2, 5, 8])
customer_satisfaction['high'] = fuzz.trimf(customer_satisfaction.universe, [5, 10, 10])

# Kural tabanını oluşturma
rules = [
    ctrl.Rule(service_speed['fast'] & food_quality['excellent'], customer_satisfaction['high']),
    ctrl.Rule(service_speed['slow'] & food_quality['bad'], customer_satisfaction['low']),
    ctrl.Rule(service_speed['normal'] & food_quality['good'], customer_satisfaction['medium']),
    ctrl.Rule(service_speed['fast'] & food_quality['bad'], customer_satisfaction['medium']),
    ctrl.Rule(service_speed['slow'] & food_quality['excellent'], customer_satisfaction['medium']),
    ctrl.Rule(service_speed['normal'] & food_quality['excellent'], customer_satisfaction['high']),
    ctrl.Rule(service_speed['normal'] & food_quality['bad'], customer_satisfaction['low']),
    ctrl.Rule(service_speed['fast'] & food_quality['good'], customer_satisfaction['high']),
    ctrl.Rule(service_speed['slow'] & food_quality['good'], customer_satisfaction['medium']),
]

# Kontrol sistemini oluşturma
satisfaction_ctrl = ctrl.ControlSystem(rules)
satisfaction_simulator = ctrl.ControlSystemSimulation(satisfaction_ctrl)

# SIMPLE DISPATCH for the given energy system with sector coupling, PV, and storage

# import necessary libraries

import logging
import oemof
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from oemof.solph import Sink, Source, Transformer, Bus, Flow, EnergySystem, Model
from oemof.solph.components import GenericStorage
import oemof.outputlib as outputlib

oemof.tools.logger.define_logging(logfile='oemof example.log', screen_level=logging.INFO, file_level=logging.DEBUG)

# Creating the energy system

date_time_index = pd.date_range('1/1/2018', periods=24*365, freq='H')

es = EnergySystem(timeindex=date_time_index)

filename = 'data_timeseries.csv'
data = pd.read_csv(filename, sep=",")

logging.info('Energy system created and initialized')

# Creating the necessary buses

elbus = Bus(label='electricity')
gasbus = Bus(label='gas')
thbus = Bus(label='heat')

logging.info('Necessary buses for the system created')

# Now creating the necessary components for the system

gas = Source(label='gas_com', outputs={gasbus: Flow()})

pv = Source(label='pv', outputs={elbus: Flow(nominal_value=65, fixed=True, actual_value=data['pv'])})

chp_gas = Transformer(label='chp_gas',
                      inputs={gasbus: Flow()},
                      outputs={elbus: Flow(nominal_value=55), thbus: Flow(nominal_value=55)},
                      conversion_factors={elbus: 0.3, thbus: 0.4})

el_storage = GenericStorage(label='el_storage',
                            nominal_storage_capacity=1000,
                            inputs={elbus: Flow(nominal_value=9)},
                            outputs={elbus: Flow(nominal_value=9)},
                            loss_rate=0.01,
                            initial_storage_level=0,
                            max_storage_level=0.9,
                            inflow_conversion_factor=0.9,
                            outflow_conversion_factor=0.9)

"""
COP = np.random.uniform(low=3.0, high=5.0, size=(8760,))

heat_pump = Transformer(label="heat_pump", inputs={elbus: Flow()},
                        outputs={thbus: Flow(nominal_value=20, variable_costs=10)},
                        conversion_factors={thbus: COP})
"""

cop = 3

heat_pump = Transformer(
    label='heat_pump',
    inputs={elbus: Flow()},
    outputs={thbus: Flow()},
    conversion_factors={elbus: cop})

logging.info('Necessary components created')

# Creating the demands

eldemand = Sink(label='eldemand', inputs={elbus: Flow(nominal_value=85, actual_value=data['demand_el'], fixed=True)})

thdemand = Sink(label='thdemand', inputs={thbus: Flow(nominal_value=40, actual_value=data['demand_th'], fixed=True)})


# Creating the excess sink and the shortage source

excess_el = Sink(label='excess_el', inputs={elbus: Flow()})

shortage_el = Source(label='shortage_el', outputs={elbus: Flow(variable_costs=1e20)})

# Adding all the components to the energy system

es.add(excess_el, shortage_el, thdemand, eldemand, heat_pump, el_storage, chp_gas, pv, gas, gasbus, thbus, elbus)

# Create the model for optimization and run the optimization

opt_model = Model(es)
opt_model.solve(solver='cbc')

logging.info('Optimization successful')

# Post-processing and data visualization

results_main = outputlib.processing.results(opt_model)
results_meta = outputlib.processing.meta_results(opt_model)
params = outputlib.processing.parameter_as_dict(es)

print(results_meta)

print(results_main[gasbus, chp_gas]['sequences'].head())


flows_el = pd.DataFrame(index=date_time_index)
flows_el['PV'] = results_main[pv, elbus]['sequences']
flows_el['CHP'] = results_main[chp_gas, elbus]['sequences']
flows_el['Excess'] = results_main[elbus, excess_el]['sequences']
flows_el['Shortage'] = results_main[shortage_el, elbus]['sequences']
flows_el['Total'] = flows_el.sum(axis=1)

flows_el_percentage = pd.DataFrame(index=date_time_index)
for column in flows_el.columns:
    if column != 'Total':
        flows_el_percentage[column] = flows_el[column] / flows_el['Total']

flows_el_percentage.plot.area()

plt.show()


es.results['main'] = results_main
es.results['meta'] = results_meta
es.results.keys()

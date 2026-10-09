import training
import interface

import matplotlib
matplotlib.use('TkAgg')

def main():   
    training_params = training.generate(10)
    network_params = (2, 16, 2)
    wb_params = training.train(training_params=training_params, network_params=network_params, steps=100)
    interface.run(wb_params)

main()
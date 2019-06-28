import misc
from main import main
from tools.dict_cart_prod import dict_cartesian_product
from pydoc import locate
from tools.printd import printd

params_grid = {
    "ELM": {
        "neurons": [1e3, 1e4, 1e5, 2.5e5, 5e5],
        "l2": [2.5e2, 5e2, 7.5e2,1e3]
    },
    "GP": {
        "alpha": [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],
        "sigma_0": [1e-8],
    },
    "SVR": {
        "C": [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3],  # [1e-2, 1e1]
        "epsilon": [1e-2],
        "gamma": [1e-5,1e-4,1e-3,1e-2, 1e-1,1e0],  # [1e-5, 1e-2],
        "shrinking": [True],
    },
}

model_name = "SVR"

dataset = "Ohio"
subject = "570"


def grid_search():
    """
        Run a grid search for a given model, dataset and subject, following the hyperparameters grids defined above
        :return: / prints the results of the grid search
    """
    Model = locate("models." + model_name + "." + model_name)
    combinations = dict_cartesian_product(params_grid[model_name])
    params_l = [{x: vals[i] for i, x in enumerate(params_grid[model_name].copy())} for vals in combinations]

    for params in params_l:
        printd(dataset, subject, model_name, params)
        main(dataset=dataset,
             subject=subject,
             Model=Model,
             params=params,
             ph=misc.ph,
             eval="valid",
             print=True,
             plot=False,
             save=False,
             excel_file="gs.xlsx")


if __name__ == "__main__":
    grid_search()

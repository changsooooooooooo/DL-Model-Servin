import torch
import yaml
import sqlalchemy
import pandas as pd
import numpy as np
from flask import request, Flask

from model.model import Model

with open("../config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.SafeLoader)
trained_model = Model()
trained_model.load_state_dict(torch.load("../model/coin-model.pth"))
trained_model.eval()
app = Flask(__name__)


def connect(user, password, db, host, port):
    url = 'postgresql://{}:{}@{}:{}/{}'
    url = url.format(user, password, host, port, db)
    return sqlalchemy.create_engine(url, client_encoding='utf8')


def preprocessing(dataframe):
    data = np.array(dataframe.iloc[:, 1:], dtype=np.float32)
    data = torch.from_numpy(data)
    return data.view(1, 1, data.size()[0], data.size()[1])


@app.route("/selltime", methods=["GET"])
def predictSellTime():
    parameter_dict = request.args.to_dict()

    if not len(parameter_dict):
        return "Require Coin Name"

    with torch.no_grad():
        table_name = parameter_dict["tablename"]
        coin_name = parameter_dict["coinname"]
        db = connect(config["db"]["username"],
                     config["db"]["password"],
                     config["db"]["dbname"],
                     config["db"]["host"],
                     config["db"]["port"])
        df = pd.read_sql("select * from {} where coin_name={}"
                         .format(table_name, coin_name),
                         db)
        inputs = preprocessing(df)
        outputs = trained_model(inputs)[0].item()
        return dict(coin_name=coin_name, sell_time=outputs)


if __name__ == '__main__':
    app.run(host=config["app"]["host"], port=config["app"]["port"])

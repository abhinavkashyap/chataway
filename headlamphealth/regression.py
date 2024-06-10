import pandas as pd
from sklearn.linear_model import LinearRegression
from typing import Any
from sklearn.model_selection import train_test_split


class PredictAnxiety:
    def __init__(
        self, sample_data_filename: str, **linear_regression_kwargs: dict[str, Any]
    ):
        """Predict the anxiety of patient given
        other factors like sleep diet social
        values and others

        Parameters
        ----------
        sample_data_filename : str
            Sample data filename where
            simpler data is stored.
        linear_regression_kwargs: dict[str, Any]
            Keyword arguments that will be passed
            to linear regression anaylsis
        """
        self.sample_data_filename = sample_data_filename

        # This is the excel sheet that has been given to me
        self.df = pd.read_excel(self.sample_data_filename)

        self.linear_regression_kwargs = linear_regression_kwargs or {}

        # diet and social seem to have NaNs.
        # This can be horribly wrong. But we can replace them
        # with average values for now for demonstration purposes
        # This has to be checked with other domain experts
        for col in ["diet", "social"]:
            self.df[col] = self.df[col].fillna(self.df[col].mean())

        self.model = LinearRegression(**self.linear_regression_kwargs)

        # Make the train and the test data
        # Get the splits for the train and the test data
        train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=1729)

        self.train_x = train_df[["sleep", "diet", "social", "exercise"]]
        self.train_y = train_df["anxious"]

        self.test_x = test_df[["sleep", "diet", "social", "exercise"]]
        self.test_y = test_df["anxious"]

        self.trained_model = self.fit(self.train_x, self.train_y)

    def fit(self, X, y):
        """Fits a linear Regression Model

        Parameters
        ----------
        X : pandas.DataFrame
            DataFrame containing the dependent variables
        y : pandas.DataFrame
            DataFrame containing the independent variable
            This is mostly anxiety for now

        Returns
        -------
        LinearRegression
            The model that is fit with the values
        """
        return self.model.fit(X, y)

    def predict(self, sleep: int, diet: int, social: int, exercise: int) -> float:
        """Predict the anxiety that the person might have based on the journal.
        This works for a single entry and single input. Does not suppoer batch input yet

        Parameters
        ----------
        sleep : int
            The value for sleep from the app
            This is one of [10,20,30]
        diet : str
            The value of diet from the app
        social : str
            The value of social from the app
        exercise : str
            The value of exercise from the app

        Returns
        -------
        float
            The predicted value of anxiety
        """

        # return the first value.
        return self.trained_model.predict([[sleep, diet, social, exercise]])[0]

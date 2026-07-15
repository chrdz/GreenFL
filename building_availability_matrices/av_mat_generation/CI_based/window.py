import os
import random
from datetime import timedelta, datetime
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('pgf') # use the pgf backend
import cvxpy as cp
import numpy as np
from matplotlib.colors import ListedColormap
import seaborn as sns
from av_mat_generation.CI_based.greedy import GreedyProblem
from math import ceil

from pathlib import Path
import pandas as pd

# Directory of this file: .../building_availability_matrices/av_mat_generation/CI_based
_THIS_DIR = Path(__file__).resolve().parent

# Root of building_availability_matrices
_BUILDING_AVAIL_DIR = _THIS_DIR.parent.parent

# Directory with the historical CSVs
HISTORICAL_DATA_DIR = _BUILDING_AVAIL_DIR / "historical_data"

LIST_COLORS = ["blue", "green", "orange", "red", "purple", "pink", "yellow"]
COUNTRIES = [
    "Ireland",
    "Germany",
    "Great Britain",
    "France",
    "Sweden",
    "Finland",
    "Belgium",
    "Brazil",
    "Denmark",
    "Estonia",
    "Spain",
    "Hungary",
    "Singapore",
    "Italy",
    "Japan",
    "South Africa",
    "Uruguay",
    "Croatia",
]
MAIN_FOLDER = "availability_matrices/av-mat-NEW"


def load_data(countries=None):
    """
    Loads the CI data in df_dict a dictionary where key=country, value=dataframe of CI data.
    Returns: df_dict
    The columns of each dataframe are datetime, CI_direct, CI_LA.
    The unit of the CI data is: gCO2eq/kWh.
    """

    # prepare links to the data csv files
    folder = HISTORICAL_DATA_DIR
    _paths = {
        "Germany": os.path.join(folder, "DE_2022_hourly.csv"),
        "Austria": os.path.join(folder, "AT_2022_hourly.csv"),
        "Ireland": os.path.join(folder, "IE_2022_hourly.csv"),
        "Great Britain": os.path.join(folder, "GB_2022_hourly.csv"),
        "France": os.path.join(folder, "FR_2022_hourly.csv"),
        "Sweden": os.path.join(folder, "SE-SE3_2022_hourly.csv"),
        "Finland": os.path.join(folder, "FI_2022_hourly.csv"),
        "Belgium": os.path.join(folder, "BE_2022_hourly.csv"),
        "Brazil": os.path.join(folder, "BR_2022_hourly.csv"),
        "Denmark": os.path.join(folder, "DK_2022_hourly.csv"),
        "Estonia": os.path.join(folder, "EE_2022_hourly.csv"),
        "Spain": os.path.join(folder, "ES_2022_hourly.csv"),
        "Hungary": os.path.join(folder, "HU_2022_hourly.csv"),
        "Singapore": os.path.join(folder, "SG_2022_hourly.csv"),
        "Japan": os.path.join(folder, "JP_2022_hourly.csv"),
        "South Africa": os.path.join(folder, "ZA_2022_hourly.csv"),
        "Peru": os.path.join(folder, "PE_2022_hourly.csv"),
        "Croatia": os.path.join(folder, "HR_2022_hourly.csv"),
    }
    # loading the data in a pandas dataframe
    df_dict = {}
    usecols = [
        "Datetime (UTC)",
        "Carbon Intensity gCO₂eq/kWh (direct)",
        "Carbon Intensity gCO₂eq/kWh (LCA)",
    ]

    for key in countries if countries else _paths.keys():
        try:
            df_dict[key] = pd.read_csv(
                _paths[key], usecols=usecols, parse_dates=["Datetime (UTC)"]
            )
            df_dict[key] = df_dict[key].rename(
                columns={
                    "Datetime (UTC)": "datetime",
                    "Carbon Intensity gCO₂eq/kWh (direct)": "CI_direct",
                    "Carbon Intensity gCO₂eq/kWh (LCA)": "CI_LCA",
                }
            )
        except KeyError:
            print(f"Data for country {key} not present")
            continue

    return df_dict

def load_data_custom(nb_countries=7):
    """
    Loads the CI data in df_dict a dictionary where key=country, value=dataframe of CI data.
    Returns: df_dict
    The columns of each dataframe are datetime, CI_direct, CI_LA.
    The unit of the CI data is: gCO2eq/kWh.
    """

    folder = Path("historical_data")
    usecols = [
        "Datetime (UTC)",
        "Carbon Intensity gCO₂eq/kWh (direct)",
        "Carbon Intensity gCO₂eq/kWh (LCA)",
    ]
    df_dict = dict()

    for p in folder.iterdir():
        if p.name not in ['CR_2022_hourly.csv', 'AU-TAS_2022_hourly.csv', 'AU-SA_2022_hourly.csv', 'CA-QC_2022_hourly.csv']:
            val = pd.read_csv(
                p,
                usecols=usecols + ["Country", "Zone Id"],
                parse_dates=["Datetime (UTC)"],
            ).rename(columns={
                "Datetime (UTC)": "datetime",
                "Carbon Intensity gCO₂eq/kWh (direct)": "CI_direct",
                "Carbon Intensity gCO₂eq/kWh (LCA)": "CI_LCA"
            })
            country = val["Country"].iloc[0]+ "-" + val["Zone Id"].iloc[0]
            df_dict[country] = val[["datetime", "CI_direct", "CI_LCA"]]

    # extract sublist of countries based on clients number:
    countries_list = list(df_dict.keys())
    rng = random.Random(42)  # reproducible
    rng.shuffle(countries_list)
    countries_number=nb_countries
    selected_countries = countries_list[:countries_number]
    df_dict = {k: df_dict[k] for k in selected_countries}

    return df_dict


class Window:

    def __init__(
        self,
        start_time=datetime(2022, 1, 1, 0, 0),
        random_start=False,
        n_rounds=100,
        countries=COUNTRIES,
        out_folder=MAIN_FOLDER,
        verbose=False,
        custom_client_list=False
    ):
        self.n_rounds = n_rounds  # number of FL training rounds
        self.countries = countries
        self.verbose = verbose
        self.custom_client_list=custom_client_list

        self.n_clients = len(countries)
        self._dfs = self._get_data()

        # add here change with random choice TODO
        self.start_time = start_time

        self.start_time, self.end_time = self._get_start_end_time(
            random_start=random_start
        )

        self.out_folder = out_folder + "/" + self.start_time.strftime("%Y-%m-%d_%H")
        self.window_list_hours = self.get_datetime_seq(self.countries[0])

        self._create_folders()
        self.CI_matrix = self.get_CI_matrix()
        self.GHG_matrix = self.get_GHG_matrix()

    def _get_data(self):
        """
        Fetch only necessary countries from the data folder
        """
        if self.custom_client_list==False:
            return load_data(self.countries)
        else:
            df_dict=load_data_custom(self.custom_client_list)
            self.countries = list(df_dict.keys())
            return df_dict

    def _get_start_end_time(self, random_start):
        """
        Return start time and end time
        """
        tot_list_hours = next(iter(self._dfs.values()))[
            "datetime"
        ].to_list()  # list of hour timestamps

        start_time = (
            random.choice(tot_list_hours[: -self.n_rounds + 1])
            if random_start
            else self.start_time
        )
        return start_time, start_time + timedelta(hours=self.n_rounds - 1)

    def get_datetime_seq(self, country):
        """
        Returns array of datetime values of country between start_date and end_date.
        Start_date and end_date are datetime objects.
        Country is a string.
        """
        df_to_plot = self._dfs[country][
            self._dfs[country]["datetime"].between(self.start_time, self.end_time)
        ]
        return df_to_plot["datetime"].values

    def _create_folders(self):
        """
        Create folders and subfolders where the images and csv files will be saved
        """
        newpath_base = self.out_folder
        # newpath_list = [newpath_base, newpath_base+'/ft', newpath_base+'/no-ft']
        newpath_list = [newpath_base]
        for newpath in newpath_list:
            if not os.path.exists(newpath):
                os.makedirs(newpath)

    def get_CI_seq(self, country):
        """
        Returns array of CI values of country between start_date and end_date.
        Start_date and end_date are datetime objects.
        Country is a string.
        """
        df_to_plot = self._dfs[country][
            self._dfs[country]["datetime"].between(self.start_time, self.end_time)
        ]
        return df_to_plot["CI_direct"].values

    def get_CI_matrix(self):
        """
        This matrix contains CI values in gCO2eq/kWh.
        """

        CI_matrix = pd.DataFrame(
            index=self.countries, columns=[i for i in range(self.n_rounds)]
        )
        for country in self.countries:
            CI_values = self.get_CI_seq(country)
            CI_matrix.loc[country, :] = CI_values
        return CI_matrix

    def get_GHG_matrix(self):
        """
        We suppose that the power draw for all clients is constant and equal. At the moment we take the constant **300 Watts**.
        The formula for carbon footprint is: power (W) * duration (hour) * carbon intensity (gCO2/kWh) * 1000
        It contains the values of carbon fooprint of sustaining a certain power during ``round_duration`` hours
        """
        power = 300 / 1000  # in kiloWatts
        round_duration = 1  # in hours: this is duration of a FL training round
        GHG_matrix = self.CI_matrix * power * round_duration / 1000
        # GHG_mat_np = GHG_matrix.to_numpy()
        return GHG_matrix

    def plot_raw_CI(self):
        """
        Plot the raw CI data of all countries over the current time window.
        """
        fig = plt.figure(figsize=(8, 4))
        
        colorblind_palette = sns.color_palette("bright")

        for country_idx, country in enumerate(self.countries):

            df_country = self._dfs[country]
            df_to_plot = df_country[
                df_country["datetime"].between(self.start_time, self.end_time)
            ]

            # plot:
            # plt.plot(
            #     df_to_plot["datetime"].values,
            #     df_to_plot["CI_direct"].values,
            #     label=country,
            #     color=LIST_COLORS[country_idx],
            # )            
            plt.plot(
                [i for i in range(len(df_to_plot["datetime"].values))],
                df_to_plot["CI_direct"].values/1000,
                label=country,
                # color=LIST_COLORS[country_idx],
                color=colorblind_palette[country_idx % len(colorblind_palette)],
            )
            # plt.title("Carbon Intensity time evolution", fontsize=14)
            plt.legend(bbox_to_anchor=(0.5, 1.25), ncol=4, loc='upper center', frameon=False,fontsize=12)
            plt.grid()
            # plt.xticks(rotation=45, ha="right", fontsize=14)  # rotate x-axis labels to diagonal
            plt.xticks(rotation=0, fontsize=14)  # rotate x-axis labels to diagonal
            plt.yticks(rotation=0, fontsize=14)  # rotate x-axis labels to diagonal
            plt.ylabel("CI (kgCO2e/kWh)", fontsize=14)
            plt.xlabel("hours", fontsize=14)

        plt.savefig(
            os.path.join(self.out_folder, "raw_CI_data.png"), bbox_inches="tight", dpi=300
        )
        plt.show()

    def plot_mean_CI(self):
        """
        Display the mean CI value for each country in between these two dates, and the global mean over all countries.
        """

        mean_list = []
        for country in self.countries:
            CI_values = self.get_CI_seq(country)
            mean_list.append(CI_values.mean())
        # dict_countries_CI = dict(zip(self.countries, mean_list))

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))
        plt.plot(self.countries, mean_list, "o")
        plt.title("Mean CI over the chosen period")
        plt.ylabel("CI (gCO2eq/kWh)")
        plt.xticks(rotation=45, ha="right")  # rotate x-axis labels to diagonal
        plt.grid()

        global_mean_CI = np.array(mean_list).mean()
        ax.axhline(y=global_mean_CI, color="r", linestyle="-", label="mean CI")
        median_CI = np.median(np.array(mean_list))
        ax.axhline(y=median_CI, color="g", linestyle="-", label="median CI")
        ax.legend()
        plt.savefig(self.out_folder + "/means.png", bbox_inches="tight")
        plt.show()

    def plot_availability_heatmap(self, similarity_matrix, key_word):
        """
        Plot heatmap of availability matrix (countries x datetime list).
        Green: available, Red: not available.
        """
        plt.figure(figsize=(7, 2))
        ax = plt.subplot()
        # cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256)
        # sns.heatmap(similarity_matrix.astype(int), annot=False, fmt='d', cbar=False, cmap=cmap, linewidths=0.5, linecolor='white', ax=ax) # create heatmap
        cmap = ListedColormap(["red", "green"])
        sns.heatmap(
            similarity_matrix,
            annot=False,
            fmt="d",
            cbar=False,
            cmap=cmap,
            linewidths=0.5,
            linecolor="white",
            vmin=0,
            vmax=1,  # Set limits to ensure correct color mapping
            ax=ax,
        )
        if not isinstance(similarity_matrix.columns[0], np.int64):
            plt.xticks(rotation=45, ha="right")  # rotate x-axis labels to diagonal
        xticks = ax.get_xticks()
        xticks = xticks[::2]
        ax.set_xticks(xticks)  # set new xticks
        plt.title(key_word)
        # plt.savefig(self.out_folder + "/" + key_word + ".png", bbox_inches="tight")
        plt.show()

    def save_availability_matrix(self, key_word, availability_matrix):
        """
        Save the availability matrix (dataframe) given as input.
        """
        dict_cols = dict(
            zip([i for i in range(self.n_rounds)], self.window_list_hours)
        )  # get the datetime values
        availability_matrix_to_save = availability_matrix.rename(columns=dict_cols)
        availability_matrix_to_save.to_csv(
            self.out_folder + "/av-mat_" + key_word + ".csv",
            columns=self.window_list_hours,
        )

    def _av_mat_alphaF(self, method, carbon_budget, key_word, alpha_f):
        """
        Solve optimization problem with fairness parameter alpha=0.1
        """
        if method.split("_")[0] == "cvxpy":
            return self._av_mat_alphaF_cvxpy(
                method.split("_")[1], carbon_budget, key_word, alpha_f=alpha_f
            )
        elif method == "greedy":
            return self._av_mat_alphaF_greedy(carbon_budget, key_word, alpha_f=alpha_f)

    def _av_mat_alphaF_cvxpy(
        self, solver, carbon_budget, key_word="alphaF", client_weights = np.array([0.17,0.03,0.25,0.14,0.14,0.26,0.01]),alpha_f=0.1
    ):
        """
        Solve optimization problem with fairness parameter alpha=0.1 through cvxpy package
        """
        w = np.ones(self.n_rounds)
        GHG_mat = self.GHG_matrix.to_numpy()
        coef = np.linalg.norm(GHG_mat)
        one_m_GHG_w = (np.max(GHG_mat) - GHG_mat) @ np.diag(w)

        x = cp.Variable(GHG_mat.shape, integer=True)
        objective = cp.Maximize(
            cp.sum(cp.power(cp.sum(cp.multiply(one_m_GHG_w, x), axis=1), alpha_f))
        )
        constraints = [0 <= x, x <= 1, cp.sum(cp.multiply(GHG_mat, x)) <= carbon_budget]
        prob = cp.Problem(objective, constraints)

        if solver == "mosek":
            result = prob.solve(solver=cp.MOSEK, verbose=self.verbose)
        elif solver == "scip":
            result = prob.solve(
                solver=cp.SCIP,
                verbose=self.verbose,
                scip_params={"limits/totalnodes": 1000},
            )
        availability_matrix = np.array(x.value, dtype=np.int8)

        availability_df = pd.DataFrame(
            availability_matrix,
            index=self.countries,
            columns=[i for i in range(self.n_rounds)],
        )

        self.plot_availability_heatmap(availability_df, key_word)
        self.save_availability_matrix(key_word, availability_df)

        return availability_df, key_word

    def _av_mat_alphaF_greedy(self, carbon_budget, key_word="alphaF", alpha_f=0.1):
        """
        Solve optimization problem with fairness parameter alpha=0.1 through greedy method
        """
        w = np.ones(self.n_rounds)
        pb = GreedyProblem(self.GHG_matrix, alpha_f, w)
        pb.greedy_optimization(carbon_budget)

        availability_matrix = np.array(pb.A, dtype=np.int8)
        availability_df = pd.DataFrame(
            availability_matrix,
            index=self.countries,
            columns=[i for i in range(self.n_rounds)],
        )

        self.plot_availability_heatmap(availability_df, key_word)
        # self.save_availability_matrix(key_word, availability_df)
        return availability_df, key_word

    def _find_largest_true_index(self, availability_matrix):
        """
        Find the end of the training.
        """
        lst = availability_matrix.sum(axis=0) == 0
        largest_index = -1
        for i in range(len(lst)):
            if lst[i] == False:
                largest_index = i
        return largest_index

    def apply_FT(self, availability_df, ft=10, carbon_budget=7, key_word="alphaF-FT"):
        """
        Add a fine-tuning phase to the availablity matrix given as input.
        Firstly, the end of the training is determined as the last time at which a
        client is available. Secondly, the fine-tuning phase is allocated as
        available. Thirdly, some slots previously allocated as available are set as
        unavailable to ensure respect of the carbon budget.
        """

        availability_matrix = availability_df.to_numpy()

        idx_end_train = self._find_largest_true_index(availability_matrix)
        # print(idx_end_train)

        percentage_fine_tuning = ft / 100
        idx = int(percentage_fine_tuning * availability_matrix.shape[1])
        # print('idx: ', idx)

        GHG_values = self.GHG_matrix.to_numpy()
        GHG_countries = sum(GHG_values[:, idx_end_train + 1 - idx : idx_end_train + 1])
        # print(GHG_countries)

        remaining_budget = carbon_budget - sum(GHG_countries)
        # print('remaining budget: ', remaining_budget)

        # Set the last 10 percent to available
        availability_matrix[:, idx_end_train + 1 - idx : idx_end_train + 1] = 1

        ### Remove green slots with the highest carbon footprint so as not to exceed the carbon budget
        # We first put a mask on slots that we should not change in this step.
        mask = availability_matrix == 0
        mask[:, idx_end_train + 1 - idx :] = True

        # Due to the previous step, the schedule is emitting extra-carbon.
        carbon_to_remove = (
            sum(sum(self.GHG_matrix.to_numpy() * availability_matrix)) - carbon_budget
        )

        GHG_mat_np = self.GHG_matrix.to_numpy()

        # mask elements that unavailable, and mask last x%
        # masked elements are set with the value 999999
        GHG_mat_masked = np.ma.array(data=GHG_mat_np, mask=mask, fill_value=999999)

        # flatten the masked array, but keep the mask
        flattened = GHG_mat_masked.flatten()
        availability_flattened = availability_matrix.flatten()

        # idx_map = np.array([i for i in range(len(flattened))]).reshape(GHG_mat_np.shape)
        # print(idx_map)

        # sort in increasing order the opposite of the flattened array, masked values are at the beginning
        # the original flattened array is sorted in decreasing order, masked values at the end
        idx_sort_flattened = (-flattened).argsort()

        # apply the sorting to the flattened array
        sorted_flattened = flattened[idx_sort_flattened]

        # compute the cumulative sum
        sorted_flattened_cumsum = sorted_flattened.cumsum()
        
        if sorted_flattened_cumsum[-1] < carbon_to_remove:
            raise ValueError("carbon_to_remove exceeds total carbon available.")
    
        # we are then able to see up to which index we should remove availability
        threshold_idx = np.where(sorted_flattened_cumsum > carbon_to_remove)[0][0]

        # these are the indexes of slots that will be made non-available
        selected_idx = idx_sort_flattened[: threshold_idx + 1]

        # modification of the availability matrix
        availability_flattened[selected_idx] = 0
        final_availability_matrix = availability_flattened.reshape(GHG_mat_np.shape)

        availability_df = pd.DataFrame(
            final_availability_matrix,
            index=self.countries,
            columns=[i for i in range(self.n_rounds)],
        )
        self.plot_availability_heatmap(availability_df, key_word)
        self.save_availability_matrix(key_word, availability_df)

        return availability_df

    def get_av_mat(
        self,
        method="cvxpy_mosek",
        key_word=None,
        fine_tuning=False,
        ft=10,
        carbon_budget=7,
        CO2saving=None, # percentage of saved carbon-footprint
        alpha_f=0.1
    ):
        if CO2saving is not None:
            # print(self.GHG_matrix.to_numpy())
            total_GHG = sum(sum(self.GHG_matrix.to_numpy()))
            print("Initial GHG: ", total_GHG)
            if CO2saving == 0.0:
                carbon_budget = ceil(total_GHG)
            else:
                carbon_budget = (1 - CO2saving) * total_GHG

        if not key_word:
            if method.split("_")[0] == "cvxpy":
                if method.split("_")[1] == "mosek":
                    key_word_NO_FT = "alphaF_cvxpy_mosek"
                    key_word_FT = "alphaF_FT_cvxpy_mosek"
                elif method.split("_")[1] == "scip":
                    key_word_NO_FT = "alphaF_cvxpy_scip"
                    key_word_FT = "alphaF_FT_cvxpy_scip"
            elif method == "greedy":
                key_word_NO_FT = "alphaF_greedy"
                key_word_FT = "alphaF_FT_greedy"
        else:
            key_word_NO_FT = key_word
            key_word_FT = key_word + f"-{ft}ft"

        av_mat_df, key_word = self._av_mat_alphaF(
            method, carbon_budget, key_word=key_word_NO_FT, alpha_f=alpha_f
        )

        print("target: ", carbon_budget)
        print(
            "result: ",
            np.sum(np.multiply(self.GHG_matrix.to_numpy(), av_mat_df.to_numpy())),
        )

        if fine_tuning == False:
            return av_mat_df
        else:
            return self.apply_FT(av_mat_df, ft, carbon_budget, key_word=key_word_FT)

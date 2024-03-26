import numpy as np
from race_lap_ngrams import RaceLapNgrams

if __name__ == "__main__":
    n = 2 # Our baseline only depends on the previous lap

    race_lap_ngrams = RaceLapNgrams(n=n, small=True)
    ngram_indices = race_lap_ngrams.ngram_indices
    ngram_df = race_lap_ngrams.df

    one_hot_start = ngram_df.columns.get_loc("Position_1.0")
    one_hot_end = ngram_df.columns.get_loc("Position_20.0")
    ngram_df["Position"] = ngram_df.iloc[:, one_hot_start:one_hot_end + 1].idxmax(axis=1).str.replace("Position_", "").str.replace(".0", "").astype(int)

    correct_positions = 0
    laptime_errors = []

    for i in ngram_indices:
        ngram = ngram_df.iloc[i: i + n]

        target_position = ngram.iloc[-1]["Position"]
        baseline_position = ngram.iloc[-2]["Position"] # previous lap position
        # baseline_position = ngram.iloc[0:-1]["Position"].mode()[0]
        # baseline_position = ngram.iloc[0]["Position"].mean().round() # average of previous n-1 lap positions
        if baseline_position == target_position:
            correct_positions += 1

        target_laptime = ngram.iloc[-1]["LapTime"]
        baseline_laptime = ngram.iloc[-2]["LapTime"] # previous lap time (lower error)
        # baseline_laptime = ngram.iloc[0:-1]["LapTime"].mean() # average of previous n-1 lap times
        laptime_errors.append(target_laptime - baseline_laptime)

    
    position_accuracy = correct_positions / len(ngram_indices)
    print(f"Position accuracy: {position_accuracy}")

    laptime_error = np.array(laptime_errors)
    laptime_mae = np.mean(np.abs(laptime_error))
    laptime_mse = np.mean(laptime_error ** 2)
    laptime_rmse = np.sqrt(laptime_mse)
    print(f"Laptime error: MAE {laptime_mae}, MSE {laptime_mse}, RMSE {laptime_rmse}")


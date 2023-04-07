import pandas as pd
import plotly.express as px

from estimagic.benchmarking.process_benchmark_results import (
    create_convergence_histories,
)
from estimagic.config import PLOTLY_TEMPLATE


def deviation_plot(
    problems,
    results,
    *,
    distance_measure="criterion",
    monotone=True,
    template=PLOTLY_TEMPLATE,
):
    """Plot average convergence of optimizers for a set of problems.

    Returns aggregated version convergence plot, showing the convergence of the
    different algorithms, averaged over a problem set. The faster a line falls, the
    faster the algorithm improved on average.

    The x axis is the runtime_measure, which can be walltime or number of evaluations.
    The y axis is the average over the convergence measures of the problems in the set.
    Convergence can be measured by the criterion value of the particular
    time/evaluation. The convergence can be made monotone by always taking the
    best  value.

    Args:
        problems (dict): estimagic benchmarking problems dictionary. Keys are the
            problem names. Values contain information on the problem, including the
            solution value.
        results (dict): estimagic benchmarking results dictionary. Keys are
            tuples of the form (problem, algorithm), values are dictionaries of the
            collected information on the benchmark run, including 'criterion_history'
            and 'time_history'.
        distance_measure (str): One of "criterion", "parameter_distance".
        monotone (bool): If True the best found criterion value so far is plotted.
            If False the particular criterion evaluation of that time is used.
        template (str): The template for the figure. Default is "plotly_white".

    Returns:
        plotly.Figure

    """

    df, _ = create_convergence_histories(
        problems=problems,
        results=results,
        stopping_criterion="y",
        x_precision=1e-6,
        y_precision=1e-6,
    )

    outcome = f"{'monotone_' if monotone else ''}" + distance_measure + "_normalized"
    deviations = (
        df.set_index(["problem", "algorithm", "n_evaluations"])[outcome]
        .reindex(
            pd.MultiIndex.from_product(
                [
                    df["problem"].unique(),
                    df["algorithm"].unique(),
                    range(df["n_evaluations"].min(), df["n_evaluations"].max() + 1),
                ],
                names=["problem", "algorithm", "n_evaluations"],
            )
        )
        .fillna(method="ffill")
        .reset_index()
    )
    average_deviations = (
        deviations.groupby(["algorithm", "n_evaluations"])
        .mean(numeric_only=True)[outcome]
        .reset_index()
    )
    fig = px.line(average_deviations, x="n_evaluations", y=outcome, color="algorithm")

    y_labels = {
        "criterion_normalized": "Share of Function Distance to Optimum<br>"
        "Missing From Current Criterion Value",
        "monotone_criterion_normalized": "Share of Function Distance to Optimum<br>"
        "Missing From Best So Far",
        "parameter_distance_normalized": "Share of Parameter Distance to Optimum<br>"
        "Missing From Current Parameters",
        "monotone_parameter_distance_normalized": "Share of the Parameter Distance "
        "to Optimum<br> Missing From the Best Parameters So Far",
    }
    fig.update_layout(
        xaxis_title="Number of Function Evaluations",
        yaxis_title=y_labels[outcome],
        title=None,
        height=300,
        width=500,
        margin={"l": 10, "r": 10, "t": 30, "b": 10},
        template=template,
    )

    return fig

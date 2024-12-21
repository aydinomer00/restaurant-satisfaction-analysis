from data_analysis import generate_and_analyze_data, visualize_data, create_correlation_analysis
from gui import InteractivePredictionGUI



def main():
    df = generate_and_analyze_data(n_samples=100)
    visualize_data(df)
    create_correlation_analysis(df)

    gui = InteractivePredictionGUI()
    gui.run()


if __name__ == "__main__":
    main()

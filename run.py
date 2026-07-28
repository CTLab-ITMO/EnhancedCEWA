import os
from pathlib import Path

import pandas as pd
from src import create_aggregator, logger
from src.utils import load_predictions_from_jsonl, load_predictions_from_csv
from src.config import Config


def main():
    config_path = Path(os.path.join('resources', 'config.yaml'))
    Config.load(config_path)
    config = Config.get_instance()
    aggregator = create_aggregator(config)

    annotations = pd.read_csv(config.annotations_path)
    os.makedirs(config.output_dir, exist_ok=True)

    if config.task == 'classification':
        model_predictions = load_predictions_from_csv(config.classification.predictions_path)
        aggregator.fit(annotations, model_predictions)
        cons_labels, quality, probs = aggregator.predict_consensus(annotations, model_predictions)
        scores = aggregator.score_annotators(annotations)

        pd.DataFrame({
            'consensus_label': cons_labels,
            'quality': quality
        }).to_csv(os.path.join(config.output_dir, 'consensus.csv'), index=False)
        logger.info("Classification aggregation completed")
    else:
        model_predictions = load_predictions_from_jsonl(config.detection.predictions_path)
        aggregator.fit(annotations, model_predictions, verbose=config.verbose)
        df_consensus = aggregator.predict_consensus(annotations, model_predictions, verbose=config.verbose)
        scores = aggregator.score_annotators()
        df_consensus.to_csv(os.path.join(config.output_dir, 'consensus.csv'), index=False)
        logger.info("Detection aggregation completed")

    pd.Series(scores).to_csv(os.path.join(config.output_dir, 'annotator_scores.csv'), index=False)
    logger.info(f"Results saved to {config.output_dir}")


if __name__ == '__main__':
    main()
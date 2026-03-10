import argparse
from app.config import load_config
from app.pipeline import SafetyPipeline


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--zones", default="configs/zones.yaml")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    pipeline = SafetyPipeline(cfg, zones_path=args.zones)
    pipeline.run()


if __name__ == "__main__":
    main()
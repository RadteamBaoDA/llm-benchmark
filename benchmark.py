#!/usr/bin/env python3
"""
LLM Benchmark Tool - CLI Entry Point

A comprehensive benchmarking tool for OpenAI-compatible APIs.
Supports chat, embedding, reranker, and vision models.
"""

import argparse
import asyncio
import sys
from pathlib import Path

from llm_benchmark.config import (
    BenchmarkConfig,
    load_config,
    save_default_config,
    save_default_scenario_config,
    init_default_configs,
    ScenarioConfig,
    APIConfig,
    ModelConfig,
    MockDataConfig
)
from llm_benchmark.engine import BenchmarkEngine, run_benchmark
from llm_benchmark.exporters import export_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LLM Benchmark Tool - Benchmark OpenAI-compatible APIs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with config file
  python benchmark.py --config config.yml
  
  # Run a quick test
  python benchmark.py --base-url http://localhost:8000 --model gpt-3.5-turbo --requests 50 --concurrency 5
  
  # Run all scenarios from config
  python benchmark.py --config config.yml --scenarios
  
  # Generate default config file
  python benchmark.py --init
  
  # Run and export results
  python benchmark.py --config config.yml --export markdown csv
        """
    )
    
    # Config file
    parser.add_argument(
        "--config", "-c",
        default="config.yml",
        help="Path to configuration file (default: config.yml)"
    )
    
    # Init mode
    parser.add_argument(
        "--init",
        action="store_true",
        help="Generate a default config.yml file"
    )
    
    # API settings (override config)
    parser.add_argument(
        "--base-url",
        help="API base URL (overrides config)"
    )
    parser.add_argument(
        "--api-key",
        help="API key for authentication (overrides config)"
    )
    
    # Model settings (override config)
    parser.add_argument(
        "--model",
        help="Model name to benchmark (overrides config)"
    )
    parser.add_argument(
        "--model-type",
        choices=["chat", "embed", "reranker", "vision"],
        help="Model type (overrides config)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Maximum tokens per request (overrides config)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for sampling (overrides config)"
    )
    
    # Benchmark settings
    parser.add_argument(
        "--requests", "-n",
        type=int,
        help="Total number of requests (for single run, overrides config)"
    )
    parser.add_argument(
        "--concurrency", "-j",
        type=int,
        help="Number of concurrent workers (for single run, overrides config)"
    )
    
    # Scenario mode
    parser.add_argument(
        "--scenarios", "-s",
        action="store_true",
        help="Run all scenarios defined in config"
    )
    parser.add_argument(
        "--scenario",
        help="Run a specific scenario by name"
    )
    
    # Output settings
    parser.add_argument(
        "--output-dir", "-o",
        help="Output directory for results (overrides config)"
    )
    parser.add_argument(
        "--export", "-e",
        nargs="+",
        choices=["markdown", "md", "csv", "json"],
        help="Export formats for results"
    )
    parser.add_argument(
        "--capture-responses",
        action="store_true",
        help="Capture full API responses"
    )
    
    # Display settings
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Hide progress bar"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def apply_overrides(config: BenchmarkConfig, args: argparse.Namespace) -> BenchmarkConfig:
    """Apply command line overrides to configuration."""
    # API overrides
    if args.base_url:
        config.api.base_url = args.base_url
    if args.api_key:
        config.api.api_key = args.api_key
    
    # Model overrides
    if args.model:
        config.model.name = args.model
    if args.model_type:
        config.model.type = args.model_type
    if args.max_tokens:
        config.model.max_tokens = args.max_tokens
    if args.temperature is not None:
        config.model.temperature = args.temperature
    
    # Benchmark overrides
    if args.requests:
        config.default_requests = args.requests
    if args.concurrency:
        config.default_concurrency = args.concurrency
    
    # Output overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.export:
        config.export_formats = args.export
    if args.capture_responses:
        config.capture_responses = True
    
    # Display overrides
    if args.quiet:
        config.quiet = True
    
    return config


def create_minimal_config(args: argparse.Namespace) -> BenchmarkConfig:
    """Create a minimal configuration from command line arguments."""
    return BenchmarkConfig(
        api=APIConfig(
            base_url=args.base_url or "http://localhost:8000",
            api_key=args.api_key or ""
        ),
        model=ModelConfig(
            name=args.model or "gpt-3.5-turbo",
            type=args.model_type or "chat",
            max_tokens=args.max_tokens or 32,
            temperature=args.temperature if args.temperature is not None else 0.2
        ),
        mock_data=MockDataConfig(),
        scenarios=[],
        default_requests=args.requests or 100,
        default_concurrency=args.concurrency or 10,
        capture_responses=args.capture_responses,
        output_dir=args.output_dir or "results",
        export_formats=args.export or ["markdown", "csv"],
        quiet=args.quiet
    )


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    # Handle init mode
    if args.init:
        config_path = args.config
        config_dir = Path(config_path).parent if Path(config_path).parent != Path('.') else Path('.')
        scenario_path = config_dir / "scenario.yml"
        
        if Path(config_path).exists() or scenario_path.exists():
            existing = []
            if Path(config_path).exists():
                existing.append(config_path)
            if scenario_path.exists():
                existing.append(str(scenario_path))
            print(f"⚠️  Configuration file(s) already exist: {', '.join(existing)}")
            response = input("Overwrite? [y/N]: ").strip().lower()
            if response != 'y':
                print("Aborted.")
                return 1
        
        save_default_config(config_path)
        save_default_scenario_config(str(scenario_path))
        print(f"✅ Created default configuration: {config_path}")
        print(f"✅ Created default scenarios: {scenario_path}")
        return 0
    
    # Load or create configuration
    config_path = Path(args.config)
    
    if config_path.exists():
        try:
            config = load_config(str(config_path))
            print(f"📄 Loaded configuration from: {config_path}")
        except Exception as e:
            print(f"❌ Error loading configuration: {e}")
            return 1
    else:
        # Check if minimal args are provided
        if not args.base_url:
            print(f"❌ Configuration file not found: {config_path}")
            print("   Run with --init to create a default config, or provide --base-url")
            return 1
        
        config = create_minimal_config(args)
        print("📄 Using command line configuration")
    
    # Apply command line overrides
    config = apply_overrides(config, args)
    
    # Run benchmarks
    try:
        if args.scenarios:
            # Run all scenarios
            if not config.scenarios:
                print("⚠️  No scenarios defined in configuration")
                return 1
            results = await run_benchmark(config)
        elif args.scenario:
            # Run specific scenario
            scenario = next(
                (s for s in config.scenarios if s.name == args.scenario),
                None
            )
            if not scenario:
                print(f"❌ Scenario not found: {args.scenario}")
                print(f"   Available scenarios: {[s.name for s in config.scenarios]}")
                return 1
            
            engine = BenchmarkEngine(config)
            results = [await engine.run_scenario(scenario, quiet=config.quiet)]
        else:
            # Run single benchmark
            engine = BenchmarkEngine(config)
            result = await engine.run_single(
                requests=config.default_requests,
                concurrency=config.default_concurrency,
                quiet=config.quiet
            )
            results = [result]
        
        # Export results
        if config.export_formats:
            print("\n📁 Exporting results...")
            exported = export_results(results, config.export_formats, config.output_dir)
            for fmt, path in exported.items():
                print(f"   {fmt}: {path}")
        
        print("\n✅ Benchmark completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())

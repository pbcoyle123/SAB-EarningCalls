#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local Environment Setup Script
==============================

This script sets up a complete local environment for the earnings call transcript 
processing pipeline, mimicking the Oracle Cloud VM setup process.

USAGE:
    python local_setup_main.py                    # Full setup + test run
    python local_setup_main.py --setup-only       # Just setup, no pipeline run
    python local_setup_main.py --test-only        # Just run test, assume setup done
    python local_setup_main.py --full-run         # Setup + full pipeline run

PURPOSE:
    - Test your environment setup before Oracle Cloud deployment
    - Ensure all dependencies work correctly on your local system
    - Verify pipeline execution in a clean environment
    - Debug any setup issues before VM deployment
"""

import os
import sys
import argparse
from pathlib import Path

def print_header():
    """Print setup header"""
    print("="*80)
    print("🧪 LOCAL ENVIRONMENT SETUP & TEST")
    print("   Earnings Call Transcript Processing Pipeline")
    print("   (Simulates Oracle Cloud VM Environment)")
    print("="*80)
    print()

def check_virtual_environment():
    """Check if we're in a virtual environment"""
    in_venv = (
        hasattr(sys, 'real_prefix') or 
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.environ.get('VIRTUAL_ENV') is not None
    )
    
    if in_venv:
        venv_path = os.environ.get('VIRTUAL_ENV', 'Unknown')
        print(f"✅ Running in virtual environment: {venv_path}")
    else:
        print("⚠️  Not running in a virtual environment")
        print("   Recommendation: Create a virtual environment first")
        print("   python -m venv test_env && test_env\\Scripts\\activate")
        
        response = input("\n   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("   Exiting. Please set up a virtual environment first.")
            sys.exit(1)
    print()

def run_environment_setup():
    """Run the environment setup process"""
    print("🔧 PHASE 1: ENVIRONMENT SETUP")
    print("-" * 50)
    
    # Import main.py functions
    try:
        # Add current directory to path to import main
        current_dir = Path(__file__).parent
        sys.path.insert(0, str(current_dir))
        
        from main import check_and_install_dependencies
        
        print("📦 Running dependency installation and model downloads...")
        setup_success = check_and_install_dependencies()
        
        if setup_success:
            print("\n✅ Environment setup completed successfully!")
            return True
        else:
            print("\n❌ Environment setup failed!")
            return False
            
    except ImportError as e:
        print(f"❌ Could not import main.py: {e}")
        print("   Make sure you're running this from the parallel_cpu directory")
        return False
    except Exception as e:
        print(f"❌ Environment setup error: {e}")
        return False

def run_pipeline_test(test_mode=True):
    """Run the pipeline in test or full mode"""
    mode_text = "TEST MODE" if test_mode else "FULL MODE"
    print(f"\n🚀 PHASE 2: PIPELINE EXECUTION ({mode_text})")
    print("-" * 50)
    
    try:
        import subprocess
        
        if test_mode:
            cmd = [sys.executable, "main.py", "--test", "--skip-setup"]
            print("📋 Running pipeline in test mode (SNAP company only)...")
        else:
            cmd = [sys.executable, "main.py", "--skip-setup"]
            print("📋 Running pipeline in full mode (all companies)...")
        
        print(f"   Command: {' '.join(cmd)}")
        print()
        
        # Run the pipeline
        result = subprocess.run(cmd, capture_output=False, text=True)
        
        if result.returncode == 0:
            print(f"\n✅ Pipeline execution completed successfully!")
            return True
        else:
            print(f"\n❌ Pipeline execution failed with exit code: {result.returncode}")
            return False
            
    except Exception as e:
        print(f"❌ Pipeline execution error: {e}")
        return False

def verify_outputs():
    """Verify that pipeline outputs were created"""
    print("\n🔍 PHASE 3: OUTPUT VERIFICATION")
    print("-" * 50)
    
    # Check output directories
    output_dirs = ["02", "03", "04", "05", "logs"]
    verification_passed = True
    
    for dir_name in output_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"   ✅ {dir_name}/: {file_count} files")
        else:
            print(f"   ❌ {dir_name}/: directory missing")
            verification_passed = False
    
    if verification_passed:
        print("\n✅ Output verification passed!")
        print("📁 Your local environment is ready for Oracle Cloud deployment!")
    else:
        print("\n⚠️  Some output directories are missing")
        print("   The pipeline may not have completed successfully")
    
    return verification_passed

def show_next_steps():
    """Show next steps for Oracle Cloud deployment"""
    print("\n" + "="*80)
    print("🎯 NEXT STEPS FOR ORACLE CLOUD DEPLOYMENT")
    print("="*80)
    print()
    print("Your local environment test was successful! Now you can:")
    print()
    print("1️⃣  CREATE VM PACKAGE:")
    print("   python create_oracle_package.py --test")
    print()
    print("2️⃣  UPLOAD TO ORACLE OBJECT STORAGE:")
    print("   Upload vm-0-package.zip to my-work-bucket/packages/")
    print()
    print("3️⃣  CREATE ORACLE VM:")
    print("   - Use cloud-init-earnings-pipeline.yaml as User Data")
    print("   - Set metadata: SHARD_ID=0, BUCKET_NAME=my-work-bucket")
    print()
    print("4️⃣  MONITOR RESULTS:")
    print("   - Check results/vm-0/ in Object Storage")
    print("   - Review logs for any issues")
    print()
    print("🚀 Your setup is Oracle Cloud ready!")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Local environment setup and testing for Oracle Cloud deployment',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  python local_setup_main.py                    # Full setup + test run
  python local_setup_main.py --setup-only       # Just setup, no pipeline
  python local_setup_main.py --test-only        # Just test, assume setup done  
  python local_setup_main.py --full-run         # Setup + full pipeline

PURPOSE:
  Test your complete environment setup before Oracle Cloud deployment.
  This script simulates the exact process that runs on Oracle VMs.
        """
    )
    
    parser.add_argument('--setup-only', action='store_true',
                       help='Only run environment setup, skip pipeline execution')
    parser.add_argument('--test-only', action='store_true', 
                       help='Only run pipeline test, assume setup is complete')
    parser.add_argument('--full-run', action='store_true',
                       help='Run setup + full pipeline (not test mode)')
    
    args = parser.parse_args()
    
    print_header()
    
    # Check virtual environment
    check_virtual_environment()
    
    # Determine what to run
    run_setup = not args.test_only
    run_pipeline = not args.setup_only
    test_mode = not args.full_run  # Default to test mode unless --full-run
    
    overall_success = True
    
    # Phase 1: Environment Setup
    if run_setup:
        setup_success = run_environment_setup()
        overall_success = overall_success and setup_success
        
        if not setup_success:
            print("\n❌ Setup failed - stopping here")
            sys.exit(1)
    
    # Phase 2: Pipeline Execution
    if run_pipeline:
        pipeline_success = run_pipeline_test(test_mode=test_mode)
        overall_success = overall_success and pipeline_success
        
        if pipeline_success:
            # Phase 3: Output Verification
            verify_outputs()
    
    # Summary
    print("\n" + "="*80)
    if overall_success:
        print("🎉 LOCAL ENVIRONMENT TEST COMPLETED SUCCESSFULLY!")
        show_next_steps()
    else:
        print("❌ LOCAL ENVIRONMENT TEST FAILED")
        print("   Please review the errors above and fix issues before Oracle Cloud deployment")
    print("="*80)
    
    sys.exit(0 if overall_success else 1)

if __name__ == "__main__":
    main()

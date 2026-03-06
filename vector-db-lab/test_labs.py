#!/usr/bin/env python3
"""
Test suite to verify all Vector Database labs work correctly.

This script runs each lab automatically and verifies:
1. Labs execute without errors
2. Expected completion files are created  
3. Basic functionality works as intended

Usage:
    python test_labs.py

Expected output:
    ✅ All labs passed successfully!
"""

import os
import sys
import subprocess
import time
from pathlib import Path

class LabTester:
    """Test runner for Vector Database labs"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.lab_dir = Path(__file__).parent
        self.completion_files = [
            'the_search_problem.txt',
            'embeddings_demo.txt', 
            'similarity_search.txt',
            'chromadb_vector_database.txt'
        ]
        
    def print_header(self):
        """Print test suite header"""
        print("🧪 Vector Database Labs Test Suite")
        print("=" * 50)
        print(f"Testing directory: {self.lab_dir}")
        print(f"Python version: {sys.version}")
        print("-" * 50)
        
    def cleanup_completion_files(self):
        """Remove any existing completion files to start fresh"""
        print("🧹 Cleaning up previous test runs...")
        for file in self.completion_files:
            file_path = self.lab_dir / file
            if file_path.exists():
                file_path.unlink()
                print(f"   Removed: {file}")
                
    def run_lab(self, lab_name, script_name, expected_completion_file):
        """Run a single lab and verify it completes successfully"""
        print(f"\n🔬 Testing Lab: {lab_name}")
        print(f"   Script: {script_name}")
        
        script_path = self.lab_dir / script_name
        if not script_path.exists():
            print(f"   ❌ FAILED: Script not found: {script_path}")
            self.failed += 1
            return False
            
        try:
            # Run the lab script
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=str(self.lab_dir),
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            end_time = time.time()
            
            # Check if script executed successfully
            if result.returncode != 0:
                print(f"   ❌ FAILED: Script exited with code {result.returncode}")
                print(f"   Error output: {result.stderr}")
                self.failed += 1
                return False
                
            # Check if completion file was created
            completion_path = self.lab_dir / expected_completion_file
            if not completion_path.exists():
                print(f"   ❌ FAILED: Completion file not created: {expected_completion_file}")
                self.failed += 1
                return False
                
            # Success!
            duration = end_time - start_time
            print(f"   ✅ PASSED: Completed in {duration:.1f}s")
            print(f"   Created: {expected_completion_file}")
            self.passed += 1
            return True
            
        except subprocess.TimeoutExpired:
            print(f"   ❌ FAILED: Script timed out after 120 seconds")
            self.failed += 1
            return False
        except Exception as e:
            print(f"   ❌ FAILED: Unexpected error: {e}")
            self.failed += 1
            return False
            
    def check_dependencies(self):
        """Check if required Python packages are available"""
        print("📦 Checking dependencies...")
        
        required_packages = [
            'sqlite3',  # Built-in
            'numpy',
            'sentence_transformers', 
            'chromadb'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                if package == 'sqlite3':
                    import sqlite3
                elif package == 'numpy':
                    import numpy
                elif package == 'sentence_transformers':
                    import sentence_transformers
                elif package == 'chromadb':
                    import chromadb
                print(f"   ✅ {package}")
            except ImportError:
                print(f"   ❌ {package} - MISSING")
                missing_packages.append(package)
                
        if missing_packages:
            print(f"\n💡 To install missing packages:")
            print(f"   pip install {' '.join(missing_packages)}")
            return False
        return True
        
    def run_all_tests(self):
        """Run all lab tests"""
        self.print_header()
        
        # Check dependencies first
        if not self.check_dependencies():
            print("\n❌ Cannot run tests - missing dependencies")
            return False
            
        # Clean up any previous test files
        self.cleanup_completion_files()
        
        # Define labs to test
        labs = [
            ("Search Problem", "search_problem.py", "the_search_problem.txt"),
            ("Embeddings Demo", "embeddings_demo.py", "embeddings_demo.txt"), 
            ("Similarity Search", "similarity_search.py", "similarity_search.txt"),
            ("Vector Database", "vector_database.py", "chromadb_vector_database.txt")
        ]
        
        # Run each lab
        for lab_name, script_name, completion_file in labs:
            self.run_lab(lab_name, script_name, completion_file)
            
        # Print summary
        self.print_summary()
        return self.failed == 0
        
    def print_summary(self):
        """Print test results summary"""
        print("\n" + "=" * 50)
        print("📊 TEST RESULTS SUMMARY")
        print("-" * 50)
        print(f"✅ Passed: {self.passed}")
        print(f"❌ Failed: {self.failed}")
        print(f"📁 Total: {self.passed + self.failed}")
        
        if self.failed == 0:
            print("\n🎉 All labs passed successfully!")
            print("🚀 Your vector database environment is ready to go!")
        else:
            print(f"\n⚠️  {self.failed} lab(s) failed")
            print("💡 Check the error messages above for troubleshooting")
            print("🔧 Common fixes:")
            print("   - Ensure virtual environment is activated")
            print("   - Run: pip install -r requirements.txt")
            print("   - Check Python version >= 3.8")

def main():
    """Main entry point"""
    tester = LabTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

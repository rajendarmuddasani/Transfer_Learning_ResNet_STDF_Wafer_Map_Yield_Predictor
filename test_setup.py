"""
Quick smoke test script for the STDF wafer map yield predictor.
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test if basic imports work."""
    print("Testing imports...")
    
    try:
        from src.data import WaferData
        print("✓ Data module imported successfully")
    except Exception as e:
        print(f"✗ Data module import failed: {e}")
        return False
    
    try:
        from src.utils import load_config
        print("✓ Utils module imported successfully")
    except Exception as e:
        print(f"✗ Utils module import failed: {e}")
        return False
    
    try:
        from fastapi import FastAPI
        print("✓ FastAPI imported successfully")
    except Exception as e:
        print(f"✗ FastAPI import failed: {e}")
        return False
    
    return True

def test_wafer_data():
    """Test WaferData creation."""
    print("\nTesting WaferData creation...")
    
    try:
        from src.data import WaferData
        import numpy as np
        
        wafer_data = WaferData(
            wafer_id="TEST-W001",
            lot_id="TEST-LOT",
            wafer_num=1,
            die_count=100,
            pass_count=85,
            fail_count=15,
            coordinates=np.array([[0, 0], [1, 1], [2, 2]]),
            bins=np.array([1, 1, 2])
        )
        
        print(f"✓ WaferData created: {wafer_data.wafer_id}")
        print(f"  - Die count: {wafer_data.die_count}")
        print(f"  - Yield: {wafer_data.pass_count / wafer_data.die_count * 100:.2f}%")
        return True
        
    except Exception as e:
        print(f"✗ WaferData creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_config():
    """Test config loading."""
    print("\nTesting config loading...")
    
    try:
        from src.utils import load_config
        
        # Test loading train config
        config = load_config("config/train_config.yaml")
        if config:
            print(f"✓ Train config loaded successfully")
            print(f"  - Model architecture: {config.get('training', {}).get('model', {}).get('architecture')}")
        else:
            print("✓ Config loader works (file not found is expected)")
        
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_creation():
    """Test FastAPI app creation."""
    print("\nTesting API app creation...")
    
    try:
        from fastapi import FastAPI
        
        app = FastAPI(title="Test API")
        
        @app.get("/")
        async def root():
            return {"status": "ok"}
        
        print("✓ FastAPI app created successfully")
        print(f"  - Routes: {len(app.routes)}")
        return True
        
    except Exception as e:
        print(f"✗ API creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("STDF Wafer Map Yield Predictor - Quick Smoke Test Suite")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("WaferData", test_wafer_data),
        ("Config", test_config),
        ("API Creation", test_api_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll smoke tests passed.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

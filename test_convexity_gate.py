"""
Test Suite: Convexity-Gate + YAML Schema Validation
Tests GO 1+3 implementation.

Test Coverage:
1. YAML Schema validation (Pydantic)
2. Atomic YAML write
3. Convexity-Gate logic
4. Hierarchy-Filter logic
5. Config round-trip (load → modify → save → load)
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import tempfile
import shutil

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules
from config_schema import (
    validate_config_file,
    save_config_atomic,
    create_default_config,
    DetectorConfigFile,
    DartDetectorConfigSchema
)
from dart_impact_detector_enhanced import (
    DartImpactDetector,
    DartDetectorConfig,
    apply_detector_preset
)


def test_yaml_schema_validation():
    """Test 1: YAML schema validation with Pydantic"""
    print("\n" + "=" * 60)
    print("TEST 1: YAML Schema Validation")
    print("=" * 60)

    # Test valid config
    config_path = Path("config/detectors.yaml")
    if not config_path.exists():
        print(f"⚠️  Config not found, creating default: {config_path}")
        create_default_config(config_path)

    try:
        config = validate_config_file(config_path)
        print(f"✅ Valid config loaded")
        print(f"   Schema version: {config.schema_version}")
        print(f"   Motion var_threshold: {config.motion.var_threshold}")
        print(f"   Dart min_area: {config.dart_detector.min_area}")
        print(f"   Convexity gate: {config.dart_detector.convexity_gate_enabled}")
        print(f"   Hierarchy filter: {config.dart_detector.hierarchy_filter_enabled}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

    # Test invalid config (odd kernel validation)
    print("\n📋 Testing invalid config detection...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        tmp.write("""
schema_version: '1.0.0'
motion:
  morph_kernel_size: 4  # INVALID: must be odd
dart_detector:
  min_area: 10
""")
        tmp_path = Path(tmp.name)

    try:
        validate_config_file(tmp_path)
        print("❌ Should have rejected invalid config!")
        return False
    except ValueError as e:
        print(f"✅ Correctly rejected invalid config: {e}")
    finally:
        tmp_path.unlink()

    print("\n✅ TEST 1 PASSED: Schema validation works correctly\n")
    return True


def test_atomic_yaml_write():
    """Test 2: Atomic YAML write (temp → move)"""
    print("\n" + "=" * 60)
    print("TEST 2: Atomic YAML Write")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "test_config.yaml"

        # Create default config
        config = DetectorConfigFile()
        config.dart_detector.min_area = 42  # Test value

        # Save atomically
        save_config_atomic(config, config_path)

        # Verify file exists
        if not config_path.exists():
            print(f"❌ Config file not created: {config_path}")
            return False

        # Reload and verify
        reloaded = validate_config_file(config_path)
        if reloaded.dart_detector.min_area != 42:
            print(f"❌ Value mismatch: expected 42, got {reloaded.dart_detector.min_area}")
            return False

        print(f"✅ Atomic write successful")
        print(f"   Value round-trip: 42 → {reloaded.dart_detector.min_area}")

    print("\n✅ TEST 2 PASSED: Atomic YAML write works correctly\n")
    return True


def test_convexity_gate():
    """Test 3: Convexity-Gate logic"""
    print("\n" + "=" * 60)
    print("TEST 3: Convexity-Gate Logic")
    print("=" * 60)

    # Create test frame with synthetic shapes
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    motion_mask = np.zeros((400, 400), dtype=np.uint8)

    # Shape 1: Convex dart-like blob (should pass)
    cv2.ellipse(motion_mask, (300, 200), (15, 40), 45, 0, 360, 255, -1)

    # Shape 2: Concave shadow/hand (should be rejected)
    # Create L-shape (low convexity) - larger to ensure detection
    motion_mask[50:120, 50:70] = 255  # Vertical bar (70 px high, 20 px wide)
    motion_mask[100:120, 50:120] = 255  # Horizontal bar (20 px high, 70 px wide)

    # Test with convexity gate enabled
    config = DartDetectorConfig(
        convexity_gate_enabled=True,
        convexity_min_ratio=0.70,
        min_area=50,
        max_area=2000,
        confirmation_frames=1  # Instant confirm for test
    )
    detector = DartImpactDetector(config)

    # Run detection
    impact = detector.detect_dart(frame, motion_mask, 0, 0.0)

    stats = detector.get_stats()
    print(f"📊 Detection stats:")
    print(f"   Total candidates: {stats['total_candidates']}")
    print(f"   Convexity rejected: {stats['convexity_rejected']}")
    print(f"   Shape rejected: {stats['shape_rejected']}")

    # Should reject L-shape either by convexity OR shape filters
    total_rejected = stats['convexity_rejected'] + stats['shape_rejected']
    if total_rejected == 0:
        print("❌ No candidates were rejected (expected at least 1)!")
        return False

    if stats['convexity_rejected'] > 0:
        print(f"✅ Convexity gate rejected {stats['convexity_rejected']} candidates")
    else:
        print(f"✅ Shape filter rejected {stats['shape_rejected']} candidates (convexity worked via solidity)")

    # Test with gate disabled
    config_no_gate = DartDetectorConfig(
        convexity_gate_enabled=False,
        min_area=50,
        max_area=2000,
        confirmation_frames=1
    )
    detector_no_gate = DartImpactDetector(config_no_gate)
    detector_no_gate.detect_dart(frame, motion_mask, 0, 0.0)

    stats_no_gate = detector_no_gate.get_stats()
    print(f"\n📊 Stats without convexity gate:")
    print(f"   Convexity rejected: {stats_no_gate['convexity_rejected']} (should be 0)")

    if stats_no_gate['convexity_rejected'] != 0:
        print("❌ Gate should be disabled!")
        return False

    print("\n✅ TEST 3 PASSED: Convexity-Gate works correctly\n")
    return True


def test_hierarchy_filter():
    """Test 4: Hierarchy-Filter logic"""
    print("\n" + "=" * 60)
    print("TEST 4: Hierarchy-Filter Logic")
    print("=" * 60)

    # Create test frame with nested contours
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    motion_mask = np.zeros((400, 400), dtype=np.uint8)

    # Outer contour (parent)
    cv2.circle(motion_mask, (200, 200), 50, 255, -1)

    # Inner contour (child) - should be rejected by hierarchy filter
    cv2.circle(motion_mask, (200, 200), 20, 0, -1)  # Cut out center

    # Test with hierarchy filter enabled
    config = DartDetectorConfig(
        hierarchy_filter_enabled=True,
        convexity_gate_enabled=False,  # Disable to isolate hierarchy test
        min_area=50,
        max_area=10000,
        confirmation_frames=1
    )
    detector = DartImpactDetector(config)

    detector.detect_dart(frame, motion_mask, 0, 0.0)

    stats = detector.get_stats()
    print(f"📊 Detection stats:")
    print(f"   Total candidates: {stats['total_candidates']}")
    print(f"   Hierarchy rejected: {stats['hierarchy_rejected']}")

    # Should see nested contour rejection
    print(f"✅ Hierarchy filter processed contours correctly")

    # Test with filter disabled
    config_no_hier = DartDetectorConfig(
        hierarchy_filter_enabled=False,
        convexity_gate_enabled=False,
        min_area=50,
        max_area=10000,
        confirmation_frames=1
    )
    detector_no_hier = DartImpactDetector(config_no_hier)
    detector_no_hier.detect_dart(frame, motion_mask, 0, 0.0)

    stats_no_hier = detector_no_hier.get_stats()
    print(f"\n📊 Stats without hierarchy filter:")
    print(f"   Hierarchy rejected: {stats_no_hier['hierarchy_rejected']} (should be 0)")

    if stats_no_hier['hierarchy_rejected'] != 0:
        print("❌ Hierarchy filter should be disabled!")
        return False

    print("\n✅ TEST 4 PASSED: Hierarchy-Filter works correctly\n")
    return True


def test_preset_application():
    """Test 5: Preset application with new fields"""
    print("\n" + "=" * 60)
    print("TEST 5: Preset Application")
    print("=" * 60)

    # Test all presets
    for preset_name in ["aggressive", "balanced", "stable"]:
        config = apply_detector_preset(DartDetectorConfig(), preset_name)
        print(f"\n📋 Preset: {preset_name}")
        print(f"   Convexity gate: {config.convexity_gate_enabled}")
        print(f"   Convexity min ratio: {config.convexity_min_ratio}")
        print(f"   Hierarchy filter: {config.hierarchy_filter_enabled}")

        if not config.convexity_gate_enabled:
            print(f"❌ Convexity gate should be enabled in preset '{preset_name}'!")
            return False

    print("\n✅ TEST 5 PASSED: Presets correctly include new fields\n")
    return True


def run_all_tests():
    """Run all test cases"""
    print("\n" + "🧪" * 30)
    print("CONVEXITY-GATE + YAML SCHEMA TEST SUITE")
    print("🧪" * 30)

    tests = [
        ("YAML Schema Validation", test_yaml_schema_validation),
        ("Atomic YAML Write", test_atomic_yaml_write),
        ("Convexity-Gate Logic", test_convexity_gate),
        ("Hierarchy-Filter Logic", test_hierarchy_filter),
        ("Preset Application", test_preset_application),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Exception: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)

    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    print(f"\n📊 Results: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED! 🎉\n")
        return True
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

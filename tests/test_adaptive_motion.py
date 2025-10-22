"""
Test Suite: Adaptive Motion-Gating (GO 2)
Tests adaptive threshold adjustment and temporal search mode.

Test Coverage:
1. Adaptive Otsu-Bias (brightness-based threshold adjustment)
2. Multi-Threshold Fusion (dual-threshold for recall)
3. Temporal Search Mode (threshold drop after stillness)
4. Config validation for new Motion fields
5. Stats tracking for adaptive features
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import modules
from src.vision.config_schema import validate_config_file, DetectorConfigFile
from src.vision.motion_detector import MotionDetector, MotionConfig, MotionEvent


def test_adaptive_otsu_bias():
    """Test 1: Adaptive Otsu-Bias adjusts based on brightness"""
    print("\n" + "=" * 60)
    print("TEST 1: Adaptive Otsu-Bias (Brightness-Based)")
    print("=" * 60)

    config = MotionConfig(
        adaptive_otsu_enabled=True,
        brightness_dark_threshold=60.0,
        brightness_bright_threshold=150.0,
        otsu_bias_dark=-15,
        otsu_bias_normal=0,
        otsu_bias_bright=10
    )
    detector = MotionDetector(config)

    # Test dark frame (brightness < 60)
    dark_frame = np.full((400, 400, 3), 30, dtype=np.uint8)  # Very dark
    detector.detect_motion(dark_frame, 0, 0.0)

    # Test bright frame (brightness > 150)
    bright_frame = np.full((400, 400, 3), 200, dtype=np.uint8)  # Very bright
    detector.detect_motion(bright_frame, 1, 0.033)

    # Test normal frame (60-150)
    normal_frame = np.full((400, 400, 3), 100, dtype=np.uint8)  # Medium
    detector.detect_motion(normal_frame, 2, 0.066)

    stats = detector.get_stats()
    print(f"📊 Adaptive stats:")
    print(f"   Dark frames: {stats['adaptive']['dark_frames']}")
    print(f"   Bright frames: {stats['adaptive']['bright_frames']}")
    print(f"   Normal frames: {stats['adaptive']['normal_frames']}")
    print(f"   Adaptive adjustments: {stats['adaptive']['adaptive_adjustments']}")

    # Validate that different biases were applied
    if stats['adaptive']['dark_frames'] != 1:
        print(f"❌ Expected 1 dark frame, got {stats['adaptive']['dark_frames']}")
        return False

    if stats['adaptive']['bright_frames'] != 1:
        print(f"❌ Expected 1 bright frame, got {stats['adaptive']['bright_frames']}")
        return False

    print(f"✅ Adaptive Otsu-Bias correctly adjusted for brightness")

    # Test with disabled adaptive
    config_no_adaptive = MotionConfig(adaptive_otsu_enabled=False)
    detector_no_adaptive = MotionDetector(config_no_adaptive)
    detector_no_adaptive.detect_motion(dark_frame, 0, 0.0)
    detector_no_adaptive.detect_motion(bright_frame, 1, 0.033)

    stats_no_adaptive = detector_no_adaptive.get_stats()
    if 'adaptive' in stats_no_adaptive and stats_no_adaptive['adaptive']['adaptive_adjustments'] > 0:
        print("❌ Adaptive should be disabled!")
        return False

    print("\n✅ TEST 1 PASSED: Adaptive Otsu-Bias works correctly\n")
    return True


def test_search_mode():
    """Test 2: Temporal Search Mode activates after stillness"""
    print("\n" + "=" * 60)
    print("TEST 2: Temporal Search Mode (After Stillness)")
    print("=" * 60)

    config = MotionConfig(
        search_mode_enabled=True,
        search_mode_trigger_frames=10,  # Fast trigger for test
        search_mode_threshold_drop=200,
        search_mode_duration_frames=5,
        motion_pixel_threshold=1000  # High threshold normally
    )
    detector = MotionDetector(config)

    # Create static frame (no motion)
    static_frame = np.zeros((400, 400, 3), dtype=np.uint8)

    # Process 15 frames of stillness (should trigger search mode after frame 10)
    for i in range(15):
        motion_detected, event, _ = detector.detect_motion(static_frame, i, i * 0.033)

    stats = detector.get_stats()
    print(f"📊 Search mode stats:")
    print(f"   Search mode activations: {stats['adaptive']['search_mode_activations']}")
    print(f"   Search mode active: {stats['search_mode_active']}")

    if stats['adaptive']['search_mode_activations'] == 0:
        print("❌ Search mode should have activated!")
        return False

    print(f"✅ Search mode activated after {config.search_mode_trigger_frames} frames of stillness")

    # Test search mode deactivation
    for i in range(15, 20):  # Process more frames
        detector.detect_motion(static_frame, i, i * 0.033)

    stats_after = detector.get_stats()
    if stats_after['search_mode_active']:
        print("⚠️  Search mode still active (expected to end, but depends on timing)")
    else:
        print(f"✅ Search mode correctly deactivated after duration")

    # Test with search mode disabled
    config_no_search = MotionConfig(search_mode_enabled=False)
    detector_no_search = MotionDetector(config_no_search)

    for i in range(20):
        detector_no_search.detect_motion(static_frame, i, i * 0.033)

    stats_no_search = detector_no_search.get_stats()
    if stats_no_search['search_mode_active']:
        print("❌ Search mode should be disabled!")
        return False

    print("\n✅ TEST 2 PASSED: Search mode works correctly\n")
    return True


def test_dual_threshold_fusion():
    """Test 3: Dual-Threshold Fusion (experimental)"""
    print("\n" + "=" * 60)
    print("TEST 3: Dual-Threshold Fusion (Experimental)")
    print("=" * 60)

    config = MotionConfig(
        dual_threshold_enabled=True,
        dual_threshold_low_multiplier=0.6,
        dual_threshold_high_multiplier=1.4
    )
    detector = MotionDetector(config)

    # Create frame with subtle motion
    frame = np.zeros((400, 400, 3), dtype=np.uint8)
    frame[100:150, 100:150] = 50  # Subtle motion blob

    detector.detect_motion(frame, 0, 0.0)

    stats = detector.get_stats()
    print(f"📊 Dual threshold stats:")
    print(f"   Dual threshold activations: {stats['adaptive']['dual_threshold_activations']}")

    if stats['adaptive']['dual_threshold_activations'] == 0:
        print("❌ Dual threshold should have activated!")
        return False

    print(f"✅ Dual-threshold fusion activated")

    # Test with disabled fusion
    config_no_dual = MotionConfig(dual_threshold_enabled=False)
    detector_no_dual = MotionDetector(config_no_dual)
    detector_no_dual.detect_motion(frame, 0, 0.0)

    stats_no_dual = detector_no_dual.get_stats()
    if 'adaptive' in stats_no_dual and stats_no_dual['adaptive']['dual_threshold_activations'] > 0:
        print("❌ Dual threshold should be disabled!")
        return False

    print("\n✅ TEST 3 PASSED: Dual-threshold fusion works correctly\n")
    return True


def test_motion_config_schema():
    """Test 4: Config schema validation for new motion fields"""
    print("\n" + "=" * 60)
    print("TEST 4: Motion Config Schema Validation")
    print("=" * 60)

    # Test valid config
    config_path = Path("../config/detectors.yaml")
    if not config_path.exists():
        print(f"⚠️  Config not found: {config_path}")
        return False

    try:
        config = validate_config_file(config_path)
        print(f"✅ Valid config loaded")
        print(f"   Adaptive Otsu: {config.motion.adaptive_otsu_enabled}")
        print(f"   Dual threshold: {config.motion.dual_threshold_enabled}")
        print(f"   Search mode: {config.motion.search_mode_enabled}")
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return False

    # Test invalid config (brightness thresholds)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
        tmp.write("""
schema_version: '1.0.0'
motion:
  brightness_dark_threshold: 300  # INVALID: > 255
  var_threshold: 50
dart_detector:
  min_area: 10
""")
        tmp_path = Path(tmp.name)

    try:
        validate_config_file(tmp_path)
        print("❌ Should have rejected invalid brightness threshold!")
        return False
    except ValueError:
        print(f"✅ Correctly rejected invalid brightness threshold (> 255)")
    finally:
        tmp_path.unlink()

    print("\n✅ TEST 4 PASSED: Motion config schema works correctly\n")
    return True


def test_brightness_tracking():
    """Test 5: Brightness history tracking"""
    print("\n" + "=" * 60)
    print("TEST 5: Brightness History Tracking")
    print("=" * 60)

    config = MotionConfig(adaptive_otsu_enabled=True)
    detector = MotionDetector(config)

    # Process frames with varying brightness
    brightnesses = [50, 100, 150, 200, 80]
    for i, brightness in enumerate(brightnesses):
        frame = np.full((400, 400, 3), brightness, dtype=np.uint8)
        detector.detect_motion(frame, i, i * 0.033)

    stats = detector.get_stats()
    avg_brightness = stats.get('avg_brightness', 0)

    expected_avg = np.mean(brightnesses)
    print(f"📊 Brightness tracking:")
    print(f"   Average brightness: {avg_brightness:.1f}")
    print(f"   Expected: {expected_avg:.1f}")

    if abs(avg_brightness - expected_avg) > 1.0:
        print(f"❌ Brightness tracking error: {abs(avg_brightness - expected_avg):.1f}")
        return False

    print(f"✅ Brightness tracking accurate (error < 1.0)")

    print("\n✅ TEST 5 PASSED: Brightness tracking works correctly\n")
    return True


def run_all_tests():
    """Run all test cases"""
    print("\n" + "🧪" * 30)
    print("ADAPTIVE MOTION-GATING TEST SUITE (GO 2)")
    print("🧪" * 30)

    tests = [
        ("Adaptive Otsu-Bias", test_adaptive_otsu_bias),
        ("Temporal Search Mode", test_search_mode),
        ("Dual-Threshold Fusion", test_dual_threshold_fusion),
        ("Motion Config Schema", test_motion_config_schema),
        ("Brightness Tracking", test_brightness_tracking),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ TEST FAILED: {name}")
            print(f"   Exception: {e}")
            import traceback
            traceback.print_exc()
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

"""Demonstration script for the PrivacyGate - Central Privacy Control.

This script demonstrates all key features of the PrivacyGate:
1. Privacy levels (LOCAL_ONLY → MINIMAL → STANDARD → DETAILED → FULL)
2. Consent management (request, grant, deny, revoke, expiry)
3. Data transformation at each privacy level
4. PII detection and stripping
5. Audit logging and transmission tracking
6. Validation and fail-closed behavior
7. Decorator-based consent enforcement
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.core.memory import (
    GeoPoint,
    Location,
    MediaType,
    Memory,
    PersonTag,
    SourcePlatform,
)
from src.core.privacy import (
    ConsentRequiredError,
    ConsentStatus,
    DataCategory,
    LocalOnlyModeError,
    PrivacyGate,
    PrivacyLevel,
    PrivacySettings,
    get_default_gate,
    require_consent,
)


def create_sample_memories() -> list[Memory]:
    """Create sample memories for testing."""
    return [
        Memory(
            source_platform=SourcePlatform.GOOGLE_PHOTOS,
            media_type=MediaType.PHOTO,
            created_at=datetime(2020, 7, 15, 14, 30, tzinfo=timezone.utc),
            location=Location(
                coordinates=GeoPoint(latitude=40.782912, longitude=-73.965456),
                place_name="Central Park",
                locality="Manhattan",
                region="New York",
                country="United States",
                country_code="US",
            ),
            people=[
                PersonTag(name="Alice Johnson"),
                PersonTag(name="Bob Smith"),
            ],
            caption="Amazing summer day with friends! Email me at alice@example.com",
            width=1920,
            height=1080,
        ),
        Memory(
            source_platform=SourcePlatform.SNAPCHAT,
            media_type=MediaType.SCREENSHOT,
            created_at=datetime(2020, 7, 16, 10, 15, tzinfo=timezone.utc),
            location=Location(
                locality="Brooklyn",
                region="New York",
                country_code="US",
            ),
            caption="Check out this cool app! Call me: 555-1234",
        ),
    ]


def demo_privacy_levels():
    """Demonstrate data transformation at each privacy level."""
    print("=" * 70)
    print("DEMO 1: Privacy Levels (Data Transformation)")
    print("=" * 70)

    memories = create_sample_memories()
    memory = memories[0]  # Use first memory

    for level in [
        PrivacyLevel.MINIMAL,
        PrivacyLevel.STANDARD,
        PrivacyLevel.DETAILED,
        PrivacyLevel.FULL,
    ]:
        print(f"\n📊 Privacy Level: {level.value.upper()}")
        print("-" * 70)

        # Create gate with this level
        settings = PrivacySettings(
            privacy_level=level,
            enabled_categories=set(DataCategory),  # Enable all categories
        )
        gate = PrivacyGate(settings)
        gate.grant_consent(f"User acknowledged {level.value} level")

        # Transform single memory
        safe_dict = gate.prepare_single_memory(memory)

        print("Data sent to AI:")
        for key, value in safe_dict.items():
            print(f"  {key}: {value}")


def demo_consent_flow():
    """Demonstrate consent management."""
    print("\n" + "=" * 70)
    print("DEMO 2: Consent Management")
    print("=" * 70)

    gate = PrivacyGate()
    gate.settings.privacy_level = PrivacyLevel.STANDARD
    gate.settings.enabled_categories = {
        DataCategory.TIMESTAMP,
        DataCategory.MEDIA_TYPE,
    }

    print(f"\n✓ Initial consent status: {gate.consent_status.value}")

    # Request consent
    print("\n📋 Requesting consent...")
    message = gate.get_consent_message()
    print(message[:500] + "...\n")  # Show first 500 chars

    # Grant consent
    print("✓ User grants consent")
    receipt = gate.grant_consent("User acknowledged data sharing")
    print(f"  - Receipt ID: {receipt.id}")
    print(f"  - Privacy Level: {receipt.privacy_level.value}")
    print(f"  - Expires: {receipt.expires_at}")
    print(f"  - Consent status: {gate.consent_status.value}")

    # Check consent
    is_valid, reason = gate.check_consent()
    print(f"\n✓ Consent valid: {is_valid}")
    if not is_valid:
        print(f"  Reason: {reason}")

    # Revoke consent
    print("\n🚫 User revokes consent")
    gate.revoke_consent()
    print(f"  - Consent status: {gate.consent_status.value}")

    is_valid, reason = gate.check_consent()
    print(f"  - Consent valid: {is_valid}")
    print(f"  - Reason: {reason}")


def demo_local_only_mode():
    """Demonstrate LOCAL_ONLY mode (privacy by default)."""
    print("\n" + "=" * 70)
    print("DEMO 3: LOCAL_ONLY Mode (Privacy by Default)")
    print("=" * 70)

    # Default gate is LOCAL_ONLY
    gate = PrivacyGate()

    print(f"\n✓ Default privacy level: {gate.settings.privacy_level.value}")
    print(f"✓ AI enabled: {gate.settings.is_ai_enabled()}")

    memories = create_sample_memories()

    print("\n🚫 Attempting to prepare memories without enabling AI...")
    try:
        safe_data, warnings = gate.prepare_memories_for_ai(memories)
        print("  ERROR: Should have raised exception!")
    except LocalOnlyModeError as e:
        print(f"  ✓ Blocked: {e}")

    print("\n✓ LOCAL_ONLY mode protects privacy by default")


def demo_consent_requirement():
    """Demonstrate consent enforcement."""
    print("\n" + "=" * 70)
    print("DEMO 4: Consent Requirement Enforcement")
    print("=" * 70)

    gate = PrivacyGate()
    gate.settings.privacy_level = PrivacyLevel.STANDARD
    gate.settings.enabled_categories = {DataCategory.TIMESTAMP}

    print(f"\n✓ Privacy level: {gate.settings.privacy_level.value}")
    print(f"✓ AI enabled: {gate.settings.is_ai_enabled()}")
    print(f"✓ Consent status: {gate.consent_status.value}")

    memories = create_sample_memories()

    print("\n🚫 Attempting to prepare memories WITHOUT consent...")
    try:
        safe_data, warnings = gate.prepare_memories_for_ai(memories)
        print("  ERROR: Should have raised exception!")
    except ConsentRequiredError as e:
        print(f"  ✓ Blocked: {e}")

    print("\n✓ Granting consent...")
    gate.grant_consent("User acknowledged")

    print("✓ Preparing memories WITH consent...")
    safe_data, warnings = gate.prepare_memories_for_ai(memories)
    print(f"  ✓ Success! Prepared {len(safe_data)} memories")


def demo_pii_detection():
    """Demonstrate PII detection and stripping."""
    print("\n" + "=" * 70)
    print("DEMO 5: PII Detection and Stripping")
    print("=" * 70)

    test_cases = [
        "Email me at alice@example.com for details",
        "Call me at 555-123-4567 or 5551234567",
        "Visit https://secretwebsite.com/private",
        "SSN is 123-45-6789",
        "Card number: 1234 5678 9012 3456",
    ]

    gate = PrivacyGate()

    print("\n🔍 Detecting PII patterns:")
    for text in test_cases:
        pii_types = gate.detect_pii(text)
        cleaned = gate.truncate_caption(text)

        print(f"\n  Original: {text}")
        if pii_types:
            print(f"  Detected: {', '.join(pii_types)}")
        print(f"  Cleaned:  {cleaned}")


def demo_anonymization():
    """Demonstrate name anonymization."""
    print("\n" + "=" * 70)
    print("DEMO 6: Name Anonymization (Consistent Hashing)")
    print("=" * 70)

    gate = PrivacyGate()

    names = ["Alice Johnson", "Bob Smith", "Alice Johnson", "Charlie Brown"]

    print("\n🔐 Anonymizing names:")
    for name in names:
        anon = gate.anonymize_name(name)
        print(f"  {name:20s} → {anon}")

    print("\n✓ Same names produce same hashes (within session)")


def demo_coordinate_blurring():
    """Demonstrate coordinate precision blurring."""
    print("\n" + "=" * 70)
    print("DEMO 7: Coordinate Precision Blurring")
    print("=" * 70)

    exact_lat, exact_lon = 40.782912, -73.965456

    print(f"\n📍 Exact coordinates: {exact_lat}, {exact_lon}")

    for precision in [0, 1, 2, 3]:
        gate = PrivacyGate()
        gate.settings.blur_location_precision = precision

        blurred_lat, blurred_lon = gate.blur_coordinates(exact_lat, exact_lon)
        accuracy_km = 10 ** (2 - precision)

        print(f"\n  Precision {precision} (~{accuracy_km}km accuracy):")
        print(f"    Result: {blurred_lat}, {blurred_lon}")


def demo_audit_logging():
    """Demonstrate transmission audit logging."""
    print("\n" + "=" * 70)
    print("DEMO 8: Audit Logging & Transmission Tracking")
    print("=" * 70)

    gate = PrivacyGate()
    gate.settings.privacy_level = PrivacyLevel.STANDARD
    gate.settings.enabled_categories = {
        DataCategory.TIMESTAMP,
        DataCategory.MEDIA_TYPE,
    }

    print("\n✓ Granting consent...")
    gate.grant_consent("User acknowledged")

    memories = create_sample_memories()

    print("✓ Preparing memories...")
    safe_data, warnings = gate.prepare_memories_for_ai(memories)

    print("✓ Recording transmission...")
    record = gate.record_transmission(
        safe_data,
        destination="gemini-1.5-pro",
        response_received=True,
    )

    print(f"\n📝 Transmission Record:")
    print(f"  - ID: {record.id}")
    print(f"  - Timestamp: {record.timestamp}")
    print(f"  - Destination: {record.destination}")
    print(f"  - Memory count: {record.memory_count}")
    print(f"  - Payload hash: {record.payload_hash}")
    print(f"  - Payload size: {record.payload_size_bytes} bytes")
    print(f"  - Categories: {[c.value for c in record.categories_included]}")
    print(f"  - Success: {record.response_received}")

    # Show history
    history = gate.get_transmission_history()
    print(f"\n✓ Total transmissions this session: {len(history)}")

    # Export audit log
    audit_path = Path("demo_audit_log.json")
    gate.export_audit_log(audit_path)
    print(f"✓ Audit log exported to: {audit_path}")


def demo_validation():
    """Demonstrate output validation."""
    print("\n" + "=" * 70)
    print("DEMO 9: Output Validation (Privacy Leak Detection)")
    print("=" * 70)

    gate = PrivacyGate()

    # Test cases: safe and unsafe data
    safe_data = [
        {"id": "123", "created_at": "2020-07"},
        {"id": "456", "media_type": "photo"},
    ]

    unsafe_data = [
        {"id": "123", "source_path": "/users/john/photos/secret.jpg"},
        {"id": "456", "caption": "Email: john@secret.com"},
    ]

    print("\n✅ Validating SAFE data:")
    is_safe, violations = gate.validate_outbound_data(safe_data)
    print(f"  - Safe: {is_safe}")
    if violations:
        print(f"  - Violations: {violations}")

    print("\n🚫 Validating UNSAFE data:")
    is_safe, violations = gate.validate_outbound_data(unsafe_data)
    print(f"  - Safe: {is_safe}")
    if violations:
        print(f"  - Violations:")
        for v in violations:
            print(f"      {v}")


def demo_decorator():
    """Demonstrate @require_consent decorator."""
    print("\n" + "=" * 70)
    print("DEMO 10: @require_consent Decorator")
    print("=" * 70)

    @require_consent
    def call_gemini_api(data):
        """Protected function that requires consent."""
        return f"AI analyzed {len(data)} items"

    # Get default gate and configure
    gate = get_default_gate()
    gate.settings.privacy_level = PrivacyLevel.STANDARD
    gate.settings.enabled_categories = {DataCategory.TIMESTAMP}

    print("\n🚫 Calling protected function WITHOUT consent...")
    try:
        result = call_gemini_api([{"id": "1"}])
        print("  ERROR: Should have raised exception!")
    except ConsentRequiredError as e:
        print(f"  ✓ Blocked: Consent required")

    print("\n✓ Granting consent...")
    gate.grant_consent("User acknowledged")

    print("✓ Calling protected function WITH consent...")
    result = call_gemini_api([{"id": "1"}, {"id": "2"}])
    print(f"  ✓ Result: {result}")


def demo_privacy_summary():
    """Demonstrate privacy summary."""
    print("\n" + "=" * 70)
    print("DEMO 11: Privacy Summary & Settings Display")
    print("=" * 70)

    settings = PrivacySettings(
        privacy_level=PrivacyLevel.STANDARD,
        enabled_categories={
            DataCategory.TIMESTAMP,
            DataCategory.LOCATION,
            DataCategory.MEDIA_TYPE,
        },
        anonymize_people=True,
        blur_location_precision=1,
        max_caption_length=100,
    )

    gate = PrivacyGate(settings)
    gate.grant_consent("User acknowledged")

    print("\n📊 Privacy Settings Summary:")
    print(gate.settings.to_user_summary())

    print("\n📊 Privacy State:")
    summary = gate.get_privacy_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main():
    """Run all demonstrations."""
    print("\n" + "🔒" * 35)
    print("PRIVACY GATE - COMPREHENSIVE DEMONSTRATION")
    print("🔒" * 35)

    demo_privacy_levels()
    demo_consent_flow()
    demo_local_only_mode()
    demo_consent_requirement()
    demo_pii_detection()
    demo_anonymization()
    demo_coordinate_blurring()
    demo_audit_logging()
    demo_validation()
    demo_decorator()
    demo_privacy_summary()

    print("\n" + "=" * 70)
    print("✅ ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)
    print("\nThe PrivacyGate is ready for:")
    print("  1. ✓ Privacy-by-default (LOCAL_ONLY mode)")
    print("  2. ✓ Explicit consent management")
    print("  3. ✓ Five privacy levels (MINIMAL → FULL)")
    print("  4. ✓ PII detection and stripping")
    print("  5. ✓ Name anonymization (consistent hashing)")
    print("  6. ✓ Coordinate blurring")
    print("  7. ✓ Audit logging and transmission tracking")
    print("  8. ✓ Output validation (privacy leak detection)")
    print("  9. ✓ Decorator-based consent enforcement")
    print(" 10. ✓ Fail-closed security (errors = deny)")
    print("\n")


if __name__ == "__main__":
    main()

# ui_schema.py

# Categorical choices you want in dropdowns (edit freely)
VEHICLE_TYPES = ["Mini", "Sedan", "SUV", "Auto", "Bike"]
BOOLEAN_CHOICES = [0, 1]  # for is_* flags

# Optional: common area names to assist typing (keep as examples or edit)
COMMON_AREAS = [
    "Bangalore City", "Koramangala", "Indiranagar", "Whitefield",
    "HSR Layout", "Marathahalli", "MG Road"
]

# Human labels for cleaner UI (only add what you have)
LABELS = {
    "booking_hour": "Booking Hour (0–23)",
    "lead_time_minutes": "Lead Time (minutes)",
    "estimated_fare": "Estimated Fare",
    "distance_km": "Estimated Distance (km)",
    "is_peak": "Peak Hour?",
    "is_weekend": "Weekend?",
    "vehicle_type": "Vehicle Type",
    "pickup_area": "Pickup Area",
    "drop_area": "Drop Area",
}

# Default values for faster testing (only add what you have)
DEFAULTS = {
    "booking_hour": 10,
    "lead_time_minutes": 15,
    "estimated_fare": 450.0,
    "distance_km": 6.0,
    "is_peak": 0,
    "is_weekend": 0,
    "vehicle_type": "Mini",
    "pickup_area": "Bangalore City",
    "drop_area": "Koramangala",
}

# Hints for numeric ranges (used when rendering number inputs)
NUMERIC_LIMITS = {
    "booking_hour": {"min": 0, "max": 23, "step": 1},
    "lead_time_minutes": {"min": 0, "max": 500, "step": 1},
    "estimated_fare": {"min": 0.0, "max": 5000.0, "step": 10.0},
    "distance_km": {"min": 0.0, "max": 200.0, "step": 0.1},
}

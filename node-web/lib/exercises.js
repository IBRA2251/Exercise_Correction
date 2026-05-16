const EXERCISE_CATALOG = {
    squat: {
        id: 'squat', name: 'Squat', type: 'single',
        primary_angle: 'left_knee', direction: 'decrease_then_increase', min_rom: 35,
        difficulty: 'Beginner', category: 'Strength',
        muscles: ['Quadriceps', 'Glutes', 'Hamstrings', 'Core'],
        description: 'A fundamental lower-body exercise that targets your quads, glutes, and core stability.',
        instructions: 'Stand with feet shoulder-width apart, lower your hips back and down as if sitting in a chair, then push back up through your heels.',
        icon: '🦵', color: '#6366f1',
    },
    pushup: {
        id: 'pushup', name: 'Push-Up', type: 'single',
        primary_angle: 'left_elbow', direction: 'decrease_then_increase', min_rom: 30,
        difficulty: 'Beginner', category: 'Strength',
        muscles: ['Chest', 'Triceps', 'Shoulders', 'Core'],
        description: 'The classic upper-body compound movement that builds chest, tricep, and shoulder strength.',
        instructions: 'Start in a plank position with hands shoulder-width apart. Lower your chest to the floor then push back up with full arm extension.',
        icon: '💪', color: '#8b5cf6',
    },
    bicep_curl: {
        id: 'bicep_curl', name: 'Bicep Curl', type: 'single',
        primary_angle: 'left_elbow', direction: 'decrease_then_increase', min_rom: 40,
        difficulty: 'Beginner', category: 'Strength',
        muscles: ['Biceps', 'Forearms', 'Brachialis'],
        description: 'An isolation exercise that targets the biceps for arm strength and definition.',
        instructions: 'Stand with arms at your sides, curl your forearms up to shoulder level by bending at the elbows, then lower slowly.',
        icon: '🏋️', color: '#a855f7',
    },
    boxing: {
        id: 'boxing', name: 'Boxing', type: 'boxing',
        primary_angle: 'right_elbow', direction: '', min_rom: 0,
        difficulty: 'Intermediate', category: 'Combat',
        muscles: ['Shoulders', 'Core', 'Arms', 'Legs'],
        description: 'High-intensity boxing drills that improve speed, coordination, and cardiovascular fitness.',
        instructions: 'Maintain a guard stance with hands at chin level, throw punches with speed and full extension, return to guard between punches.',
        icon: '🥊', color: '#f43f5e',
    },
    yoga: {
        id: 'yoga', name: 'Yoga', type: 'yoga',
        primary_angle: 'left_knee', direction: '', min_rom: 0,
        difficulty: 'Intermediate', category: 'Flexibility',
        muscles: ['Full Body', 'Core', 'Balance', 'Flexibility'],
        description: 'Yoga poses that improve flexibility, balance, and mindfulness through controlled, stable movements.',
        instructions: 'Move into the pose slowly and with control, breathe steadily, hold the position with stability, then release gently.',
        icon: '🧘', color: '#06b6d4',
    },
};

module.exports = EXERCISE_CATALOG;

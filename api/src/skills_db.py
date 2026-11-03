"""
Skills Database - Common tech skills with synonyms and categories
for fuzzy matching in the Job Recommender System.
"""

# =======================================
# SKILL SYNONYMS MAPPING
# "short form" → "full form" (normalize කරන්න)
# =======================================
SKILL_SYNONYMS = {
    # Programming Languages
    "js": "javascript",
    "ts": "typescript",
    "py": "python",
    "rb": "ruby",
    "cpp": "c++",
    "c#": "csharp",
    "c sharp": "csharp",
    "golang": "go",
    "kt": "kotlin",

    # Frameworks & Libraries
    "react.js": "react",
    "reactjs": "react",
    "react js": "react",
    "next.js": "nextjs",
    "next js": "nextjs",
    "node.js": "nodejs",
    "node js": "nodejs",
    "vue.js": "vue",
    "vuejs": "vue",
    "vue js": "vue",
    "angular.js": "angular",
    "angularjs": "angular",
    "express.js": "express",
    "expressjs": "express",
    "flask": "flask",
    "django": "django",
    "spring boot": "spring boot",
    "springboot": "spring boot",
    ".net": "dotnet",
    "dot net": "dotnet",
    "asp.net": "dotnet",

    # Data Science / ML
    "ml": "machine learning",
    "dl": "deep learning",
    "ai": "artificial intelligence",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
    "pytorch": "pytorch",
    "torch": "pytorch",
    "sklearn": "scikit-learn",
    "scikit learn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "keras": "keras",
    "opencv": "opencv",
    "huggingface": "hugging face",
    "llm": "large language model",
    "genai": "generative ai",
    "gen ai": "generative ai",

    # Cloud & DevOps
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "google cloud": "google cloud platform",
    "azure": "microsoft azure",
    "k8s": "kubernetes",
    "docker": "docker",
    "ci/cd": "cicd",
    "ci cd": "cicd",
    "devops": "devops",
    "terraform": "terraform",
    "ansible": "ansible",
    "jenkins": "jenkins",
    "github actions": "github actions",
    "gitlab ci": "gitlab ci",

    # Databases
    "sql": "sql",
    "mysql": "mysql",
    "postgres": "postgresql",
    "postgresql": "postgresql",
    "mongo": "mongodb",
    "mongodb": "mongodb",
    "redis": "redis",
    "dynamodb": "dynamodb",
    "firebase": "firebase",
    "supabase": "supabase",
    "nosql": "nosql",
    "graphql": "graphql",

    # Web Technologies
    "html5": "html",
    "css3": "css",
    "sass": "sass",
    "scss": "sass",
    "less": "less",
    "tailwind": "tailwindcss",
    "tailwind css": "tailwindcss",
    "bootstrap": "bootstrap",
    "rest api": "rest api",
    "restful": "rest api",
    "restful api": "rest api",
    "websocket": "websockets",
    "web sockets": "websockets",

    # Tools & Practices
    "git": "git",
    "github": "github",
    "gitlab": "gitlab",
    "bitbucket": "bitbucket",
    "jira": "jira",
    "agile": "agile",
    "scrum": "scrum",
    "kanban": "kanban",
    "tdd": "test driven development",
    "bdd": "behavior driven development",
    "oop": "object oriented programming",
    "solid": "solid principles",
    "design patterns": "design patterns",
    "microservices": "microservices",
    "soa": "service oriented architecture",

    # Mobile Development
    "react native": "react native",
    "rn": "react native",
    "flutter": "flutter",
    "swift": "swift",
    "swiftui": "swiftui",
    "ios": "ios development",
    "android": "android development",
    "kotlin": "kotlin",

    # Soft Skills & Other
    "pm": "project management",
    "pmp": "project management professional",
    "ui": "user interface",
    "ux": "user experience",
    "ui/ux": "ui ux design",
    "figma": "figma",
    "adobe xd": "adobe xd",
    "photoshop": "photoshop",
    "illustrator": "illustrator",

    # Music & Performing Arts
    "daw": "digital audio workstation",
    "fl studio": "fl studio",
    "ableton": "ableton live",
    "pro tools": "pro tools",
    "logic pro": "logic pro x",
    "cubase": "cubase",
    "music prod": "music production",
    "sound eng": "sound engineering",
    "vocals": "singing",
    "vocal": "singing",
    "midi": "midi programming",
    "audio mix": "audio mixing",
    "audio master": "audio mastering",
    "musicianship": "music theory",
    "sruti": "carnatic music",
    "tabla": "tabla playing",
    "veena": "veena playing",
    "guitar": "guitar playing",
    "piano": "piano playing",
    "violin": "violin playing",
    "drums": "drum playing",
    "keyboard": "keyboard playing",
    "bass guitar": "bass guitar playing",

    # Design & Visual Arts
    "pr": "premiere pro",
    "premiere": "premiere pro",
    "ae": "after effects",
    "ps": "photoshop",
    "ai": "adobe illustrator",
    "lightroom": "adobe lightroom",
    "indesign": "adobe indesign",
    "canva": "canva",
    "video edit": "video editing",
    "photo edit": "photo editing",
    "color grading": "color grading",
    "motion graphics": "motion graphics",
    "3d modeling": "3d modeling",
    "blender": "blender 3d",
    "maya": "autodesk maya",
    "autocad": "autocad",
}

# =======================================
# KNOWN SKILLS LIST
# ML matcher එකට fuzzy match කරන්න පාවිච්චි කරනවා
# =======================================
KNOWN_SKILLS = sorted(set(
    list(SKILL_SYNONYMS.keys()) + list(SKILL_SYNONYMS.values()) + [
        # Extra skills not in synonyms
        "python", "java", "javascript", "typescript", "c++", "c", "csharp",
        "go", "rust", "ruby", "php", "scala", "r", "matlab", "perl", "swift",
        "kotlin", "dart", "lua", "haskell", "elixir", "clojure",
        "react", "angular", "vue", "svelte", "nextjs", "nuxt", "gatsby",
        "django", "flask", "fastapi", "spring", "spring boot", "express",
        "nodejs", "rails", "laravel", "asp.net",
        "html", "css", "javascript", "jquery", "ajax", "webpack", "vite",
        "babel", "eslint", "prettier",
        "tensorflow", "pytorch", "scikit-learn", "keras", "opencv",
        "pandas", "numpy", "matplotlib", "seaborn", "scipy", "statsmodels",
        "machine learning", "deep learning", "natural language processing",
        "computer vision", "reinforcement learning", "generative ai",
        "data science", "data analysis", "data engineering", "data visualization",
        "big data", "hadoop", "spark", "kafka", "airflow", "dbt",
        "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
        "cassandra", "dynamodb", "firebase", "sqlite", "oracle",
        "aws", "azure", "google cloud platform", "heroku", "digitalocean",
        "docker", "kubernetes", "terraform", "ansible", "jenkins",
        "cicd", "devops", "sre", "linux", "unix", "bash", "powershell",
        "git", "github", "gitlab", "bitbucket", "svn",
        "rest api", "graphql", "grpc", "websockets", "soap",
        "microservices", "serverless", "lambda",
        "react native", "flutter", "ionic", "xamarin",
        "ios development", "android development", "mobile development",
        "agile", "scrum", "kanban", "waterfall", "lean",
        "project management", "product management", "leadership",
        "communication", "teamwork", "problem solving", "critical thinking",
        "ui ux design", "figma", "sketch", "adobe xd", "photoshop",
        "illustrator", "after effects", "premiere pro",
        "cybersecurity", "penetration testing", "ethical hacking",
        "networking", "tcp/ip", "dns", "vpn", "firewall",
        "blockchain", "solidity", "web3", "ethereum", "smart contracts",
        "iot", "embedded systems", "arduino", "raspberry pi",
        "robotics", "ros", "control systems",
        "erp", "sap", "salesforce", "hubspot", "crm",
        "power bi", "tableau", "looker", "metabase",
        "excel", "google sheets", "microsoft office",
        "technical writing", "documentation", "api documentation",

        # =============================================
        # MUSIC & PERFORMING ARTS SKILLS
        # =============================================
        "music production", "music theory", "music composition", "music arrangement",
        "singing", "vocals", "vocal training", "songwriting", "lyric writing",
        "sound engineering", "audio mixing", "audio mastering", "audio editing",
        "recording studio", "live sound", "broadcast audio",
        "digital audio workstation", "fl studio", "ableton live", "pro tools",
        "logic pro x", "cubase", "garage band", "reaper",
        "midi programming", "music sampling", "beat making",
        "guitar playing", "piano playing", "violin playing", "drum playing",
        "keyboard playing", "bass guitar playing", "tabla playing",
        "carnatic music", "western music", "classical music", "jazz", "blues",
        "music performance", "stage performance", "live performance",
        "choir conducting", "music directing", "orchestra",
        "music teaching", "music education", "sitar", "flute playing",
        "music notation", "sight reading", "ear training",
        "film scoring", "background music", "jingle production",
        "radio production", "podcast production", "audio production",
        "dj", "turntablism", "music streaming", "artist management",

        # =============================================
        # VISUAL ARTS & DESIGN SKILLS
        # =============================================
        "graphic design", "visual design", "brand design", "logo design",
        "typography", "layout design", "print design", "digital design",
        "canva", "adobe lightroom", "adobe indesign", "adobe illustrator",
        "video editing", "photo editing", "color grading", "retouching",
        "motion graphics", "animation", "2d animation", "3d animation",
        "3d modeling", "blender 3d", "autodesk maya", "cinema 4d",
        "autocad", "sketchup", "solidworks",
        "fine arts", "drawing", "painting", "sculpture", "printmaking",
        "photography", "portrait photography", "landscape photography",
        "videography", "cinematography", "film making", "documentary",
        "fashion design", "textile design", "interior design",
        "ui design", "ux design", "wireframing", "prototyping",

        # =============================================
        # EDUCATION & TEACHING
        # =============================================
        "teaching", "lecturing", "curriculum development", "lesson planning",
        "classroom management", "student assessment", "tutoring",
        "educational technology", "e-learning", "online teaching",
        "special education", "early childhood education",
        "language teaching", "esl teaching", "sinhala teaching", "english teaching",

        # =============================================
        # BUSINESS & MANAGEMENT
        # =============================================
        "accounting", "bookkeeping", "financial analysis", "financial reporting",
        "auditing", "taxation", "payroll", "budgeting", "forecasting",
        "marketing", "digital marketing", "social media marketing",
        "content marketing", "seo", "sem", "email marketing",
        "sales", "business development", "customer relations", "crm",
        "human resources", "recruitment", "talent acquisition", "performance management",
        "supply chain", "logistics", "procurement", "inventory management",
        "operations management", "quality management", "iso",
        "event management", "project coordination", "administrative",

        # =============================================
        # HEALTHCARE & MEDICINE
        # =============================================
        "nursing", "patient care", "clinical skills", "first aid",
        "pharmacy", "pharmacology", "physiotherapy", "occupational therapy",
        "medical coding", "medical transcription", "healthcare management",
        "laboratory skills", "medical laboratory", "radiology",

        # =============================================
        # LAW & LEGAL
        # =============================================
        "legal research", "contract drafting", "litigation", "legal writing",
        "corporate law", "criminal law", "family law", "intellectual property",

        # =============================================
        # HOSPITALITY & TOURISM
        # =============================================
        "hotel management", "food and beverage", "front office", "housekeeping",
        "culinary arts", "catering", "event planning", "travel management",
        "tour guiding", "customer service", "hospitality",

        # =============================================
        # LANGUAGES & COMMUNICATION
        # =============================================
        "english", "sinhala", "tamil", "mandarin", "french", "german",
        "translation", "interpretation", "content writing", "copywriting",
        "journalism", "media production", "public relations", "broadcasting",
    ]
))

# =======================================
# EXPERIENCE LEVEL KEYWORDS
# Resume/Job එකෙන් experience level detect කරන්න
# =======================================
EXPERIENCE_LEVELS = {
    "intern": 0,
    "internship": 0,
    "trainee": 0,
    "entry level": 1,
    "entry-level": 1,
    "junior": 1,
    "associate": 1,
    "fresher": 1,
    "fresh graduate": 1,
    "graduate": 1,
    "mid level": 2,
    "mid-level": 2,
    "intermediate": 2,
    "2+ years": 2,
    "3+ years": 2,
    "senior": 3,
    "sr.": 3,
    "sr ": 3,
    "5+ years": 3,
    "lead": 4,
    "team lead": 4,
    "principal": 4,
    "staff": 4,
    "architect": 4,
    "manager": 5,
    "director": 5,
    "vp": 6,
    "vice president": 6,
    "cto": 7,
    "ceo": 7,
    "head of": 5,
}


def normalize_skill(skill_text: str) -> str:
    """Skill text එක normalize කරනවා - lowercase + synonym resolve."""
    skill_lower = skill_text.lower().strip()
    return SKILL_SYNONYMS.get(skill_lower, skill_lower)


def get_all_known_skills() -> list:
    """Known skills ලිස්ට් එක return කරනවා."""
    return KNOWN_SKILLS


def detect_experience_level(text: str) -> int:
    """
    Text එකෙන් experience level detect කරනවා.
    Returns: 0 (intern) to 7 (C-level), -1 if not detected
    """
    text_lower = text.lower()
    detected_level = -1

    for keyword, level in EXPERIENCE_LEVELS.items():
        if keyword in text_lower:
            detected_level = max(detected_level, level)

    return detected_level

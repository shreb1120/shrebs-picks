# 30 MLB teams (2022-present, after Cleveland name change)
mlb_team_index = {
    'Arizona Diamondbacks': 0,
    'Atlanta Braves': 1,
    'Baltimore Orioles': 2,
    'Boston Red Sox': 3,
    'Chicago Cubs': 4,
    'Chicago White Sox': 5,
    'Cincinnati Reds': 6,
    'Cleveland Guardians': 7,
    'Colorado Rockies': 8,
    'Detroit Tigers': 9,
    'Houston Astros': 10,
    'Kansas City Royals': 11,
    'Los Angeles Angels': 12,
    'Los Angeles Dodgers': 13,
    'Miami Marlins': 14,
    'Milwaukee Brewers': 15,
    'Minnesota Twins': 16,
    'New York Mets': 17,
    'New York Yankees': 18,
    'Oakland Athletics': 19,
    'Philadelphia Phillies': 20,
    'Pittsburgh Pirates': 21,
    'San Diego Padres': 22,
    'San Francisco Giants': 23,
    'Seattle Mariners': 24,
    'St. Louis Cardinals': 25,
    'Tampa Bay Rays': 26,
    'Texas Rangers': 27,
    'Toronto Blue Jays': 28,
    'Washington Nationals': 29,
}

# Historical team name mappings (pre-2022)
mlb_team_index_pre2022 = {
    **{k: v for k, v in mlb_team_index.items() if k != 'Cleveland Guardians'},
    'Cleveland Indians': 7,
}

# pybaseball abbreviation -> full name
PYBASEBALL_TEAM_MAP = {
    'ARI': 'Arizona Diamondbacks',
    'ATL': 'Atlanta Braves',
    'BAL': 'Baltimore Orioles',
    'BOS': 'Boston Red Sox',
    'CHC': 'Chicago Cubs',
    'CHW': 'Chicago White Sox',
    'CIN': 'Cincinnati Reds',
    'CLE': 'Cleveland Guardians',
    'COL': 'Colorado Rockies',
    'DET': 'Detroit Tigers',
    'HOU': 'Houston Astros',
    'KCR': 'Kansas City Royals',
    'LAA': 'Los Angeles Angels',
    'LAD': 'Los Angeles Dodgers',
    'MIA': 'Miami Marlins',
    'MIL': 'Milwaukee Brewers',
    'MIN': 'Minnesota Twins',
    'NYM': 'New York Mets',
    'NYY': 'New York Yankees',
    'OAK': 'Oakland Athletics',
    'PHI': 'Philadelphia Phillies',
    'PIT': 'Pittsburgh Pirates',
    'SDP': 'San Diego Padres',
    'SFG': 'San Francisco Giants',
    'SEA': 'Seattle Mariners',
    'STL': 'St. Louis Cardinals',
    'TBR': 'Tampa Bay Rays',
    'TEX': 'Texas Rangers',
    'TOR': 'Toronto Blue Jays',
    'WSN': 'Washington Nationals',
}

# Reverse map: full name -> pybaseball abbreviation
FULL_TO_ABBREV = {v: k for k, v in PYBASEBALL_TEAM_MAP.items()}

# Normalize team names across data sources (Odds API, sbrscrape, etc.)
MLB_TEAM_NAME_MAP = {
    'Cleveland Indians': 'Cleveland Guardians',
    'LA Angels': 'Los Angeles Angels',
    'Los Angeles Angels of Anaheim': 'Los Angeles Angels',
    'Anaheim Angels': 'Los Angeles Angels',
    'Florida Marlins': 'Miami Marlins',
    'Tampa Bay Devil Rays': 'Tampa Bay Rays',
    'Montreal Expos': 'Washington Nationals',
}


def normalize_team_name(name, season=None):
    """Normalize a team name to the canonical full name."""
    if name in mlb_team_index:
        return name
    if name in MLB_TEAM_NAME_MAP:
        return MLB_TEAM_NAME_MAP[name]
    if name in PYBASEBALL_TEAM_MAP:
        return PYBASEBALL_TEAM_MAP[name]
    return name


def get_team_index(season=None):
    """Return the correct team index dict for a given season year."""
    if season is not None and season < 2022:
        return mlb_team_index_pre2022
    return mlb_team_index

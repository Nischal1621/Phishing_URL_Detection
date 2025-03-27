import re
import numpy as np
from urllib.parse import urlparse
import tldextract
import socket
from functools import lru_cache

# ------------------ REGEX PATTERNS ------------------ #
IP_PATTERN = re.compile(r'^\d{1,3}(?:\.\d{1,3}){3}$')
WORD_PATTERN = re.compile(r'[A-Za-z0-9]+')
SUBDOMAIN_PATTERN = re.compile(r'^w+\d*$')

@lru_cache(maxsize=10000)
def cached_gethostbyname(hostname):
    """Cache DNS lookups to avoid redundant network calls"""
    try:
        socket.gethostbyname(hostname)
        return 1
    except:
        return 0

def extract_url_features(url):
    """ Extracts 57 URL-based features from the given URL without scaling. """
    features = {f"f{i}": 0 for i in range(1, 57)}  # Initialize with default values
    parsed = urlparse(url)
    hostname = parsed.netloc
    ext = tldextract.extract(url)
    subdomain = ext.subdomain
    tld = ext.suffix
    domain_main = ext.domain

    # Length-based features
    features['f1'] = len(url)  # URL length
    features['f2'] = len(hostname)  # Hostname length
    features['f3'] = 1 if IP_PATTERN.match(hostname) else 0  # IP in hostname

    # Special character counts
    special_chars = {4: '.', 5: '-', 6: '@', 7: '?', 8: '&', 9: '|', 10: '=', 11: '_',
                     12: '~', 13: '%', 14: '/', 15: '*', 16: ':', 17: ',', 18: ';', 19: '$'}
    for i, char in special_chars.items():
        features[f"f{i}"] = url.count(char)

    # Other URL structure features
    features['f20'] = url.count('%20') + url.count(' ')  # Spaces
    features['f21'] = url.lower().count("www")
    features['f22'] = url.lower().count(".com")
    features['f23'] = url.lower().count("http")
    features['f24'] = url.count("//")
    features['f25'] = 1 if parsed.scheme.lower() == 'https' else 0  # HTTPS present

    # Digit ratios
    digits_url = sum(c.isdigit() for c in url)
    features['f26'] = digits_url / len(url) if len(url) else 0
    digits_hostname = sum(c.isdigit() for c in hostname)
    features['f27'] = digits_hostname / len(hostname) if len(hostname) else 0
    features['f28'] = 1 if "xn--" in hostname else 0  # Punycode in hostname
    features['f29'] = 1 if parsed.port else 0  # Port present in URL

    # TLD presence
    features['f30'] = 1 if tld and (tld in parsed.path) else 0
    features['f31'] = 1 if tld and (tld in subdomain) else 0

    # Subdomain analysis
    features['f32'] = 1 if subdomain and subdomain.lower() != "www" and SUBDOMAIN_PATTERN.match(subdomain) else 0
    features['f33'] = len(subdomain.split('.')) if subdomain else 0
    features['f34'] = 1 if '-' in domain_main else 0  # Prefix-suffix in domain

    # Vowel ratio in domain name
    vowels = sum(1 for c in domain_main.lower() if c in 'aeiou')
    features['f35'] = 1 if domain_main and (vowels / len(domain_main)) < 0.3 else 0

    # Shortening services detection
    shortening_services = {'bit.ly', 'tinyurl.com', 'goo.gl', 'ow.ly', 't.co', 'is.gd', 'buff.ly', 'adf.ly'}
    features['f36'] = 1 if hostname.lower() in shortening_services else 0

    # Suspicious file extensions
    suspicious_exts = {'.exe', '.js', '.txt'}
    features['f37'] = 1 if any(parsed.path.lower().endswith(ext) for ext in suspicious_exts) else 0

    # Word-based features
    words_url = WORD_PATTERN.findall(url)
    features['f38'] = len(words_url)  # Total words in URL
    features['f39'] = max((len(w) for w in words_url), default=0)  # Longest word in URL
    features['f40'] = min((len(w) for w in words_url), default=0)  # Shortest word in URL
    features['f41'] = len(WORD_PATTERN.findall(hostname))  # Words in hostname
    features['f42'] = len(WORD_PATTERN.findall(parsed.path))  # Words in path

    # Average word length features
    features['f43'] = sum(len(w) for w in words_url) / len(words_url) if words_url else 0
    features['f44'] = sum(len(w) for w in WORD_PATTERN.findall(hostname)) / len(WORD_PATTERN.findall(hostname)) if WORD_PATTERN.findall(hostname) else 0
    features['f45'] = sum(len(w) for w in WORD_PATTERN.findall(parsed.path)) / len(WORD_PATTERN.findall(parsed.path)) if WORD_PATTERN.findall(parsed.path) else 0

    # Phish-related hints
    sensitive_words = {"login", "signin", "verify", "account", "update", "secure", "confirm", "bank", "paypal", "ebay", "admin", "security", "password"}
    features['f46'] = sum(url.lower().count(word) for word in sensitive_words)

    # Brand presence
    brands = {"google", "facebook", "amazon", "paypal", "apple", "microsoft", "ebay"}
    features['f47'] = 1 if any(brand in domain_main.lower() for brand in brands) else 0
    features['f48'] = 1 if any(brand in subdomain.lower() for brand in brands) else 0
    features['f49'] = 1 if any(brand in parsed.path.lower() for brand in brands) else 0

    # DNS resolution & Suspicious TLD detection
    features['f50'] = cached_gethostbyname(hostname)
    features['f51'] = 1 if tld.lower() in {"tk", "ml", "ga", "cf", "gq"} else 0

    # Length of domain name
    features['f52'] = len(domain_main)
    # Length of top-level domain (TLD)
    features['f53'] = len(tld) if tld else 0
    # Presence of numeric-only domain name
    features['f54'] = 1 if domain_main.isdigit() else 0
    # Number of query parameters
    features['f55'] = len(parsed.query.split('&')) if parsed.query else 0
    # Presence of fragment identifier (#)
    features['f56'] = 1 if '#' in parsed.fragment else 0
    

    # Convert dictionary to NumPy array
    feature_values = np.array([features[f"f{i}"] for i in range(1, 57)]).reshape(1, -1)

    return feature_values  # Return raw feature values (not scaled)

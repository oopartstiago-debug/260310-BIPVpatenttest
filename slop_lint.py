#!/usr/bin/env python3
"""AI-slop lint for this repo: markdown/HTML text, HTML/CSS styling, Python, and repo layout.

Usage:
    python slop_lint.py            # report
    python slop_lint.py --strict   # exit 1 if any HIGH finding
    python slop_lint.py FILE...    # lint specific files only

Advisory by design. Every rule here is a heuristic; read the hit before acting on it.
"""
import colorsys
import os
import re
import sys
from collections import Counter
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- text rules
EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF☀-➿⭐✅❌⚠️\U0001F525✨]"
)
BOLD_LABEL_BULLET = re.compile(r"^\s*(?:[-*]|\d+\.)\s+\*\*[^*]{1,40}\*\*\s*[:：—-]", re.M)
NOT_X_BUT_Y = re.compile(r"(?:가|이|은|는)\s*아니라|it'?s not [^.]{3,40}, it'?s\b", re.I)
CLOSING_FORMULA = re.compile(
    r"정리하면|한\s?문장으로|한\s?줄로|결론적으로|요약하면|다시 말해|한 마디로|In conclusion|In summary|"
    r"Key takeaways?|Let'?s dive|엔지니어용 한 줄"
)
PUFF_WORDS = re.compile(
    r"정직|진짜|완전(?:히)?|핵심|유일|근본|견고|튼튼|엄격|획기적|혁신적|강력한|원활한|포괄적|"
    r"\b(?:robust|seamless|leverage|comprehensive|delve|cutting-edge|game-changing|crucial|pivotal)\b",
    re.I,
)
DECOR_MARKS = re.compile(r"[★※①②③④⑤→↔⇒▶◆]")
FAKE_PRECISION = re.compile(r"\d+\.\d{2,}\s?%|\b\d+\.\d{3,}\b")
MD_HEADER = re.compile(r"^#{1,6}\s", re.M)
MD_BOLD = re.compile(r"\*\*[^*\n]+\*\*")
MD_TABLE_ROW = re.compile(r"^\|", re.M)

# ---------------------------------------------------------------- css rules
HEX = re.compile(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{3})\b")
FONT_FAMILY = re.compile(r"(?:font-family|--[\w-]*(?:font|sans|serif|mono)[\w-]*)\s*:\s*([^;}]+)", re.I)
GENERIC_FONTS = ("inter", "roboto", "space grotesk", "pretendard", "arial", "helvetica neue", "instrument serif")
TAILWIND_HEX = {"#8b5cf6", "#6366f1", "#a855f7", "#7c3aed", "#f97316", "#64748b", "#3b82f6", "#10b981"}


def hue_sat_lum(hexstr):
    h = hexstr.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
    return hh * 360, ss, ll


class TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.skip = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.skip:
            self.skip -= 1

    def handle_data(self, data):
        if not self.skip:
            self.parts.append(data)


def html_text(src):
    p = TextExtractor()
    p.feed(src)
    return "\n".join(p.parts)


FENCE = re.compile(r"```.*?```", re.S)
OPT_OUT = "slop-lint: off"


def lint_text(src, is_md):
    if is_md:
        src = FENCE.sub("", src)  # code blocks quote patterns on purpose
    words = max(1, len(src.split()))
    kchars = max(1.0, len(src) / 1000)
    f = []

    def add(sev, rule, n, note):
        if n:
            f.append((sev, rule, n, note))

    add("HIGH", "em-dash", len(re.findall("—", src)), f"{len(re.findall('—', src)) / kchars:.1f}/1k chars; use 쉼표·마침표·괄호")
    add("HIGH", "emoji", len(EMOJI.findall(src)), "✅⚠️💡 등은 텍스트로 대체")
    add("HIGH", "not-X-but-Y", len(NOT_X_BUT_Y.findall(src)), "'X가 아니라 Y' 대구법 반복")
    add("MED", "closing-formula", len(CLOSING_FORMULA.findall(src)), "'정리하면/한 문장으로' 류 마무리 공식")
    add("MED", "puff-words", len(PUFF_WORDS.findall(src)), "자기평가 수식어(정직/진짜/완전/핵심/유일…)")
    add("MED", "decor-marks", len(DECOR_MARKS.findall(src)), "★ ① → 등 장식 기호")
    add("LOW", "fake-precision", len(FAKE_PRECISION.findall(src)), "소수 2자리 % 또는 3자리 이상 소수")
    if is_md:
        add("HIGH", "bold-label-bullet", len(BOLD_LABEL_BULLET.findall(src)), "'- **라벨**: 설명' 패턴")
        bold = len(MD_BOLD.findall(src))
        add("MED", "bold-density", bold if bold / words > 0.03 else 0, f"{bold} bold / {words} words (>3%)")
        heads = len(MD_HEADER.findall(src))
        add("LOW", "header-density", heads if words / max(1, heads) < 60 else 0, f"{heads} headers / {words} words")
    return f


def lint_css(src):
    f = []
    css = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", src, re.S | re.I)) or src
    hexes = [h.lower() for h in HEX.findall(css)]
    purple = [h for h in hexes if 250 <= hue_sat_lum(h)[0] <= 290 and hue_sat_lum(h)[1] > 0.3]
    terracotta = [h for h in hexes if 12 <= hue_sat_lum(h)[0] <= 32 and hue_sat_lum(h)[1] > 0.5 and hue_sat_lum(h)[2] < 0.6]
    dark_ground = [h for h in hexes if hue_sat_lum(h)[2] < 0.1]
    cream_ground = [h for h in hexes if hue_sat_lum(h)[2] > 0.93 and 30 <= hue_sat_lum(h)[0] <= 70 and hue_sat_lum(h)[1] > 0.1]
    fonts = " ".join(FONT_FAMILY.findall(css)).lower()
    generic = [g for g in GENERIC_FONTS if g in fonts]
    serif_display = bool(re.search(r"newsreader|playfair|instrument serif|fraunces|crimson|source serif|georgia", fonts))

    def add(sev, rule, n, note):
        if n:
            f.append((sev, rule, n, note))

    add("HIGH", "purple-accent", len(set(purple)), f"{sorted(set(purple))[:4]} (indigo/violet = Tailwind 기본값 유산)")
    add("HIGH", "generic-font", len(generic), f"{generic} (Pretendard는 Inter 파생)")
    if dark_ground and purple:
        add("HIGH", "cliche-1:dark+purple", 1, "근흑 배경 + 보라 강조 = 1세대 AI 대시보드 룩")
    if cream_ground and serif_display and terracotta:
        add("HIGH", "cliche-2:cream+serif+terracotta", 1, "크림 배경 + 세리프 제목 + 테라코타 강조 = 2세대 AI 룩 (Anthropic frontend-design skill 명시)")
    add("MED", "tailwind-hex", len(TAILWIND_HEX.intersection(hexes)), f"{sorted(TAILWIND_HEX.intersection(hexes))}")
    add("MED", "backdrop-filter", len(re.findall(r"backdrop-filter", css)), "글래스모피즘, 2개 초과면 과다")
    add("MED", "uppercase-eyebrow", len(re.findall(r"text-transform\s*:\s*uppercase", css)), "ALL-CAPS 아이브로우 라벨 (템플릿 크롬)")
    add("MED", "infinite-animation", len(re.findall(r"animation\s*:[^;]*infinite", css)), "장식용 무한 애니메이션(로고 흔들기 등)")
    add("MED", "reveal-on-scroll", 1 if ("IntersectionObserver" in src and re.search(r"opacity\s*:\s*0", css)) else 0, "스크롤 페이드업 리빌")
    add("LOW", "big-radius", len(re.findall(r"border-radius\s*:\s*(?:1[2-9]|[2-9]\d)px", css)), "12px 이상 라운드 카드")
    add("LOW", "pill-radius", len(re.findall(r"border-radius\s*:\s*999px", css)), "999px 필 형태")
    add("LOW", "3-4-col-grid", len(re.findall(r"repeat\(\s*[34]\s*,", css)), "동일 카드 3·4열 그리드")
    text = html_text(src)
    add("MED", "middle-dot-sep", len(re.findall(r"\s·\s", text)), "'A · B · C' 구분자")
    add("MED", "arrow-glyph", len(re.findall("→", text)), "→ 링크/설명 화살표")
    add("MED", "numbered-eyebrow", len(re.findall(r"\b0[1-9]\s*·", text)), "'01 ·' 번호 아이브로우 (순서 정보가 없는데 번호)")
    return f


def lint_py(src):
    f = []
    n = len(EMOJI.findall(src))
    if n:
        f.append(("MED", "emoji-in-code", n, "UI 문자열/주석의 이모지"))
    tw = TAILWIND_HEX.intersection(h.lower() for h in HEX.findall(src))
    if tw:
        f.append(("MED", "tailwind-hex", len(tw), f"{sorted(tw)}"))
    if "plotly_dark" in src:
        f.append(("LOW", "plotly_dark-template", src.count("plotly_dark"), "라이트 테마 앱에 남은 다크 템플릿 상수"))
    narr = len(re.findall(r"#\s*(?:import|load|initialize|set up|define)\s+(?:necessary|the|all)\b", src, re.I))
    if narr:
        f.append(("LOW", "narrating-comment", narr, "코드를 다시 말하는 주석"))
    return f


def lint_repo(root):
    f = []
    top = os.listdir(root)
    verify = [x for x in top if x.startswith("verify_") and x.endswith(".py")]
    results = [x for x in top if x.endswith(".json") and ("result" in x or "summary" in x or "metrics" in x)]
    versioned = [x for x in top if re.search(r"_v\d+\b", x)]
    docx_dup = [x for x in top if x.endswith(".docx") and x[:-5] + ".md" in top]
    html = [x for x in top if x.endswith(".html")]
    sources = {}
    for x in top:
        if x.endswith((".py", ".html", ".md", ".toml")):
            try:
                sources[x] = open(os.path.join(root, x), encoding="utf-8", errors="ignore").read()
            except OSError:
                pass
    # A screen counts as referenced only from code or docs; a comment in a sibling HTML file is not a consumer.
    orphan = [
        h for h in html
        if not any(h in body for name, body in sources.items() if not name.endswith(".html"))
    ]
    if len(verify) > 5:
        f.append(("MED", "root-verify-scripts", len(verify), "verify_*.py 를 verify/ 또는 tests/ 로, 1회성은 삭제"))
    if len(results) > 5:
        f.append(("MED", "root-result-dumps", len(results), "*_results.json 을 results/ 로 또는 gitignore"))
    if versioned:
        f.append(("LOW", "version-suffix-files", len(versioned), f"_vN 접미사: git이 버전 관리. {versioned[:5]}"))
    if docx_dup:
        f.append(("LOW", "docx-duplicates", len(docx_dup), f"{docx_dup} (md의 생성물, 커밋 불필요)"))
    if orphan:
        f.append(("HIGH", "orphan-html", len(orphan), f"어디서도 참조되지 않음: {orphan}"))
    return f


SEV_ORDER = {"HIGH": 0, "MED": 1, "LOW": 2}


def main(argv):
    strict = "--strict" in argv
    files = [a for a in argv if not a.startswith("--")]
    if not files:
        files = sorted(
            x for x in os.listdir(ROOT)
            if x.endswith((".md", ".html", ".py")) and x != os.path.basename(__file__)
        )
    total = Counter()
    for name in files:
        path = os.path.join(ROOT, name) if not os.path.isabs(name) else name
        try:
            src = open(path, encoding="utf-8").read()
        except (OSError, UnicodeDecodeError):
            continue
        findings = []
        if OPT_OUT in src[:400]:
            print(f"\n{name}\n  skip  opted out ({OPT_OUT})")
            continue
        if name.endswith(".md"):
            findings = lint_text(src, True)
        elif name.endswith(".html"):
            findings = lint_text(html_text(src), False) + lint_css(src)
        elif name.endswith(".py"):
            findings = lint_py(src)
        findings.sort(key=lambda t: (SEV_ORDER[t[0]], -t[2]))
        if findings:
            print(f"\n{name}")
            for sev, rule, n, note in findings:
                total[sev] += 1
                print(f"  {sev:<4} {rule:<32} {n:>4}  {note}")
    repo = lint_repo(ROOT)
    if repo:
        print("\n[repo layout]")
        for sev, rule, n, note in sorted(repo, key=lambda t: SEV_ORDER[t[0]]):
            total[sev] += 1
            print(f"  {sev:<4} {rule:<32} {n:>4}  {note}")
    print(f"\nHIGH {total['HIGH']}  MED {total['MED']}  LOW {total['LOW']}")
    return 1 if (strict and total["HIGH"]) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

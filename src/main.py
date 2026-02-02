import os
import json
import csv
from datetime import datetime, timedelta
from mkdocs.utils import get_relative_url

def define_env(env):
    @env.macro
    def load_class_data():
        data_path = os.path.join('docs', 'data', 'class_data.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            class_data = json.load(f)
        return class_data

    @env.macro
    def course_info():
        return load_class_data()['course']

    @env.macro
    def instructor_info():
        return load_class_data()['instructor']

    @env.macro
    def content_links():
        content_data = load_class_data()['content']
        content_links = {}
        canvas_base = course_info()['canvas']
        repo_base = course_info()['repo']
        for k, v in content_data.items():
            if any(x in k for x in ("PA", "GP")):
                content_links[k] = f"{canvas_base}/assignments/{v}"
            elif any(x in k for x in ("HW", "CP")):
                content_links[k] = f"{canvas_base}/quizzes/{v}"
            elif any(x in k for x in ("Demo", )):
                content_links[k] = f"{repo_base}{v}"
            else:
                content_links[k] = v
        return content_links

    @env.macro
    def get_links(keys):
        return [content_links().get(k) for k in keys]
    
    @env.filter
    def url(target: str) -> str:
        # compute a URL for `target` relative to the *current* page
        current = getattr(env.page, "url", "")
        return get_relative_url(target, current)
    
    # Load the course_calendar.csv file once at startup
    csv_path = os.path.join('docs', 'data', 'course_calendar.csv')
    course_calendar = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            course_calendar.append({
                'date': row['date'],
                'label': row['label'],
                'title': row['title'],
                'week': row['week']
            })
    @env.macro
    def course_calendar_table(offset_week=1):
        output = []
        prev_week_no = None
        first_date_week = None
        for row in course_calendar:
            print(row)
            current_date = datetime.strptime(row['date'], "%Y-%m-%d")
            monday_date = current_date - timedelta(days=current_date.weekday())
            week_no = row['week']
            # Always link to Notes if Week 1 or week has started
            if row['label'] == 'BRK':
                header_title = row['title']
            elif week_no == '1' or datetime.now() >= monday_date and week_no != prev_week_no:
                header_title = f'<a href="notes/week_{week_no}">Week {week_no}</a>'
            else:
                header_title = f'Week {week_no}'
            if week_no != prev_week_no:
                first_date_week = current_date
                if prev_week_no is not None:
                    output.append("</tbody></table></div>")
                output.append(f"""
<div class="cal-table-wrapper">
<table class="cal-table">
    <colgroup>
        <col style="width: 20%;">
        <col style="width: 20%;">
        <col style="width: 80%;">
    </colgroup>
    <thead>
        <tr class="cal-row">
            <th colspan="3">{header_title}</th>
        </tr>
    </thead>
    <tbody>
""")
                prev_week_no = week_no
            label = row.get('label', '')
            title = row.get('title', '')
            badge_class = {
                "LECT": "blue",
                "DEMO": "green",
                "PART": "purple",
                "LAB": "purple",
                "PROJ": "gray",
                "HMWK": "yellow",
                "PROG": "orange",
                "QUIZ": "red",
                "EXAM": "red",
                "NO": "green"
            }.get(label, "red" if "Due" in title else "black" if label else "")
            label_html = f'<span class="md-cal-badge md-cal-badge-{badge_class}">{label}</span>' if label else ""
            # Handle multiple links/titles
            #links = row.get('link', '').split(';') if row.get('link') else []
            titles = title.split(';')
            links = get_links(titles)
            title_links = []
            for i, t in enumerate(titles):
                if i < len(links) and links[i]:
                    title_links.append(f'<a href="{links[i].strip()}">{t.strip()}</a>')
                else:
                    title_links.append(t.strip())
            title_html = '<br/>'.join(title_links)
            if row['label'] != 'BRK':
                output.append(f"""
    <tr class="cal-row">
        <td style="text-align: center">{current_date.strftime("%a, %b %d")}</td>
        <td style="text-align: center">{label_html}</td>
        <td style="padding-left: 4%">{title_html}</td>
    </tr>
""")
        output.append("</tbody></table></div>")
        return "\n".join(output)

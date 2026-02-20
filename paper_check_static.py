import argparse
import json
import os

from paper_check import ChecklistEvaluator


def run_static_checks(evaluator):
    """Run static checks only and return report dict."""
    evaluator.check_title_presence()
    evaluator.check_section_titles_descriptive()
    evaluator.check_section_titles_capitalization()
    evaluator.check_section_buffer()
    evaluator.static_check_figures()
    evaluator.static_check_numbers_spelling()
    evaluator.static_check_informal_words()
    evaluator.static_check_italics_usage()
    return evaluator.report


def has_static_failures(report):
    for _, check_data in report.items():
        if isinstance(check_data, dict) and check_data.get("result") is False:
            return True
    return False


def render_markdown(report):
    markdown_output = "# Paper Static Check Report\n\n"
    markdown_output += "## Summary\n\n"

    total_checks = 0
    passed_checks = 0
    for _, check_data in report.items():
        if isinstance(check_data, dict) and "result" in check_data:
            total_checks += 1
            if check_data["result"] is True:
                passed_checks += 1

    if total_checks > 0:
        pass_rate = (passed_checks / total_checks) * 100
        markdown_output += f"- **Pass Rate**: {pass_rate:.1f}% ({passed_checks}/{total_checks} checks passed)\n\n"

    markdown_output += "## Detailed Results\n\n"
    for check_name, check_data in report.items():
        if not (isinstance(check_data, dict) and "result" in check_data):
            continue

        status = "✅ PASS" if check_data["result"] is True else "❌ FAIL"
        check_title = check_data.get("check", check_name)
        markdown_output += f"### {status}: {check_title}\n\n"
        markdown_output += f"**Details**: {check_data.get('detail', 'No detailed information')}\n\n"

        failed_content = check_data.get("failed_content")
        if failed_content:
            markdown_output += "**Error content**:\n\n"
            if isinstance(failed_content, list):
                for i, item in enumerate(failed_content, 1):
                    markdown_output += f"{i}. {item}\n"
            elif isinstance(failed_content, dict):
                for key, value in failed_content.items():
                    markdown_output += f"- **{key}**:\n"
                    if isinstance(value, list):
                        for i, item in enumerate(value, 1):
                            markdown_output += f"  {i}. {item}\n"
                    else:
                        markdown_output += f"  {value}\n"
            else:
                markdown_output += f"{failed_content}\n"
            markdown_output += "\n"

    return markdown_output


def main():
    parser = argparse.ArgumentParser(description="Run static paper checks without changing paper_check.py behavior.")
    parser.add_argument("--latex", required=True, help="Path to LaTeX source file or directory")
    parser.add_argument("--api_key", required=False, help="Optional API key passthrough")
    parser.add_argument("--output-json", required=False, help="Optional path to write JSON report")
    parser.add_argument("--output-md", required=False, help="Optional path to write markdown report")
    parser.add_argument("--fail-on-static-error", action="store_true", help="Exit code 1 when static checks fail")
    args = parser.parse_args()

    # paper_check.py always loads PDF in __init__. Monkey patch to avoid PDF dependency in static-only runs.
    ChecklistEvaluator.load_pdf = lambda self: ""
    evaluator = ChecklistEvaluator("__STATIC_MODE_NO_PDF__", args.latex, args.api_key)

    report = run_static_checks(evaluator)
    markdown_output = render_markdown(report)

    if args.output_md:
        md_path = args.output_md
    else:
        md_path = "paper_static_quality_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown_output)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nMarkdown report saved to: {os.path.abspath(md_path)}")

    if args.fail_on_static_error and has_static_failures(report):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import os
import time

from .first_interaction import add_comment, get_first_interaction_response, get_relevant_labels
from .utils import GITHUB_API_URL, GITHUB_GRAPHQL_URL, MAX_PR_CHARACTERS, Action, get_completion

# Constants
SUMMARY_START = (
    "## 🛠️ PR Summary\n\n<sub>Made with ❤️ by [Ultralytics Actions](https://github.com/ultralytics/actions)<sub>\n\n"
)


def generate_merge_message(pr_summary=None, pr_credit=None, pr_url=None):
    """Generates a motivating thank-you message for merged PR contributors."""
    messages = [
        {
            "role": "system",
            "content": "You are an Ultralytics AI assistant. Generate inspiring, appreciative messages for GitHub contributors.",
        },
        {
            "role": "user",
            "content": (
                f"Write a warm thank-you comment for the merged PR {pr_url} by {pr_credit}. "
                f"Context:\n{pr_summary}\n\n"
                f"Start with an enthusiastic note about the merge, incorporate a relevant inspirational quote from a historical "
                f"figure, and connect it to the PR's impact. Keep it concise yet meaningful, ensuring contributors feel valued."
            ),
        },
    ]
    return get_completion(messages)


def post_merge_message(event, summary, pr_credit):
    """Posts thank you message on PR after merge."""
    pr_url = f"{GITHUB_API_URL}/repos/{event.repository}/pulls/{event.pr['number']}"
    comment_url = f"{GITHUB_API_URL}/repos/{event.repository}/issues/{event.pr['number']}/comments"
    message = generate_merge_message(summary, pr_credit, pr_url)
    event.post(comment_url, json={"body": message})


def generate_issue_comment(pr_url, pr_summary, pr_credit, pr_title=""):
    """Generates personalized issue comment based on PR context."""
    # Extract repo info from PR URL (format: api.github.com/repos/owner/repo/pulls/number)
    repo_parts = pr_url.split("/repos/")[1].split("/pulls/")[0] if "/repos/" in pr_url else ""
    owner_repo = repo_parts.split("/")
    repo_name = owner_repo[-1] if len(owner_repo) > 1 else "package"

    messages = [
        {
            "role": "system",
            "content": "You are an Ultralytics AI assistant. Generate friendly GitHub issue comments. No @ mentions or direct addressing.",
        },
        {
            "role": "user",
            "content": f"Write a GitHub issue comment announcing a potential fix for this issue is now merged in linked PR {pr_url} by {pr_credit}\n\n"
            f"PR Title: {pr_title}\n\n"
            f"Context from PR:\n{pr_summary}\n\n"
            f"Include:\n"
            f"1. An explanation of key changes from the PR that may resolve this issue\n"
            f"2. Credit to the PR author and contributors\n"
            f"3. Options for testing if PR changes have resolved this issue:\n"
            f"   - If the PR mentions a specific version number (like v8.0.0 or 3.1.0), include: pip install -U {repo_name}>=VERSION\n"
            f"   - Also suggest: pip install git+https://github.com/{repo_parts}.git@main\n"
            f"   - If appropriate, mention they can also wait for the next official PyPI release\n"
            f"4. Request feedback on whether the PR changes resolve the issue\n"
            f"5. Thank 🙏 for reporting the issue and welcome any further feedback if the issue persists\n\n",
        },
    ]
    return get_completion(messages)


def generate_pr_summary(repository, diff_text):
    """Generates a concise, professional summary of a PR using OpenAI's API for Ultralytics repositories."""
    if not diff_text:
        diff_text = "**ERROR: DIFF IS EMPTY, THERE ARE ZERO CODE CHANGES IN THIS PR."
    messages = [
        {
            "role": "system",
            "content": "You are an Ultralytics AI assistant skilled in software development and technical communication. Your task is to summarize GitHub PRs from Ultralytics in a way that is accurate, concise, and understandable to both expert developers and non-expert users. Focus on highlighting the key changes and their impact in simple, concise terms.",
        },
        {
            "role": "user",
            "content": f"Summarize this '{repository}' PR, focusing on major changes, their purpose, and potential impact. Keep the summary clear and concise, suitable for a broad audience. Add emojis to enliven the summary. Reply directly with a summary along these example guidelines, though feel free to adjust as appropriate:\n\n"
            f"### 🌟 Summary (single-line synopsis)\n"
            f"### 📊 Key Changes (bullet points highlighting any major changes)\n"
            f"### 🎯 Purpose & Impact (bullet points explaining any benefits and potential impact to users)\n"
            f"\n\nHere's the PR diff:\n\n{diff_text}",
        },
    ]
    reply = get_completion(messages, temperature=1.0)
    if len(diff_text) == MAX_PR_CHARACTERS:
        reply = "**WARNING ⚠️** this PR is very large, summary may not cover all changes.\n\n" + reply
    return SUMMARY_START + reply


def update_pr_description(event, new_summary, max_retries=2):
    """Updates PR description with new summary, retrying if description is None."""
    description = ""
    url = f"{GITHUB_API_URL}/repos/{event.repository}/pulls/{event.pr['number']}"
    for i in range(max_retries + 1):
        description = event.get(url).json().get("body") or ""
        if description:
            break
        if i < max_retries:
            print("No current PR description found, retrying...")
            time.sleep(1)

    # Check if existing summary is present and update accordingly
    start = "## 🛠️ PR Summary"
    if start in description:
        print("Existing PR Summary found, replacing.")
        updated_description = description.split(start)[0] + new_summary
    else:
        print("PR Summary not found, appending.")
        updated_description = description + "\n\n" + new_summary

    # Update the PR description
    event.patch(url, json={"body": updated_description})


def label_fixed_issues(event, pr_summary):
    """Labels issues closed by PR when merged, notifies users, and returns PR contributors."""
    query = """
query($owner: String!, $repo: String!, $pr_number: Int!) {
    repository(owner: $owner, name: $repo) {
        pullRequest(number: $pr_number) {
            closingIssuesReferences(first: 50) { nodes { number } }
            url
            title
            body
            author { login, __typename }
            reviews(first: 50) { nodes { author { login, __typename } } }
            comments(first: 50) { nodes { author { login, __typename } } }
            commits(first: 100) { nodes { commit { author { user { login } }, committer { user { login } } } } }
        }
    }
}
"""
    owner, repo = event.repository.split("/")
    variables = {"owner": owner, "repo": repo, "pr_number": event.pr["number"]}
    response = event.post(GITHUB_GRAPHQL_URL, json={"query": query, "variables": variables})
    if response.status_code != 200:
        return None  # no linked issues

    try:
        data = response.json()["data"]["repository"]["pullRequest"]
        comments = data["reviews"]["nodes"] + data["comments"]["nodes"]
        token_username = event.get_username()  # get GITHUB_TOKEN username
        author = data["author"]["login"] if data["author"]["__typename"] != "Bot" else None
        pr_title = data.get("title", "")

        # Get unique contributors from reviews and comments
        contributors = {x["author"]["login"] for x in comments if x["author"]["__typename"] != "Bot"}

        # Add commit authors and committers that have GitHub accounts linked
        for commit in data["commits"]["nodes"]:
            commit_data = commit["commit"]
            for user_type in ["author", "committer"]:
                if user := commit_data[user_type].get("user"):
                    if login := user.get("login"):
                        contributors.add(login)

        contributors.discard(author)
        contributors.discard(token_username)

        # Write credit string
        pr_credit = ""  # i.e. "@user1 with contributions from @user2, @user3"
        if author and author != token_username:
            pr_credit += f"@{author}"
        if contributors:
            pr_credit += (" with contributions from " if pr_credit else "") + ", ".join(f"@{c}" for c in contributors)

        # Generate personalized comment
        comment = generate_issue_comment(
            pr_url=data["url"], pr_summary=pr_summary, pr_credit=pr_credit, pr_title=pr_title
        )

        # Update linked issues
        for issue in data["closingIssuesReferences"]["nodes"]:
            number = issue["number"]
            # Add fixed label
            event.post(f"{GITHUB_API_URL}/repos/{event.repository}/issues/{number}/labels", json={"labels": ["fixed"]})

            # Add comment
            event.post(f"{GITHUB_API_URL}/repos/{event.repository}/issues/{number}/comments", json={"body": comment})

        return pr_credit
    except KeyError as e:
        print(f"Error parsing GraphQL response: {e}")
        return None


def remove_pr_labels(event, labels=()):
    """Removes specified labels from PR."""
    for label in labels:  # Can be extended with more labels in the future
        event.delete(f"{GITHUB_API_URL}/repos/{event.repository}/issues/{event.pr['number']}/labels/{label}")


def generate_unified_pr_response(event):
    """Generate PR summary, labels, and first comment in a single OpenAI call with JSON structured output."""
    print("🔧 Starting unified PR response generation...")
    pr_data = event.get_repo_data(f"pulls/{event.pr['number']}")
    available_labels = event.get_repo_data("labels")
    label_descriptions = {label["name"]: label.get("description", "") for label in available_labels}
    print(f"📊 Found {len(available_labels)} available labels")

    # Remove mutually exclusive labels and inappropriate labels
    for label in {
        "help wanted",
        "TODO",
        "research",
        "non-reproducible",
        "popular",
        "invalid",
        "Stale",
        "wontfix",
        "duplicate",
        "question",  # Remove question for PRs
    }:
        label_descriptions.pop(label, None)

    # Add "Alert" to available labels if not present
    if "Alert" not in label_descriptions:
        label_descriptions["Alert"] = (
            "Potential spam, abuse, or illegal activity including advertising, unsolicited promotions, malware, phishing, crypto offers, pirated software or media, free movie downloads, cracks, keygens or any other content that violates terms of service or legal standards."
        )

    diff = event.get_pr_diff()
    username = pr_data["user"]["login"]
    title = pr_data["title"]
    body = pr_data.get("body") or ""  # Fix: Handle None body
    print(f"👤 PR Author: @{username}")
    print(f"📝 PR Title: {title[:50]}...")
    print(f"📄 Diff length: {len(diff)} chars")
    print(f"📋 Body length: {len(body)} chars")  # Debug body

    # JSON schema for structured output
    json_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "PRAnalysisResponse",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "PR summary with sections: ### 🌟 Summary, ### 📊 Key Changes, ### 🎯 Purpose & Impact",
                    },
                    "labels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of most relevant label names",
                    },
                    "first_comment": {
                        "type": "string",
                        "description": "Welcome comment for first-time PR with checklist and guidance + PR-specific notes for {PR name}",
                    },
                },
                "required": ["summary", "labels", "first_comment"],
                "additionalProperties": False,
            },
        },
    }

    org_name, repo_name = event.repository.split("/")
    repo_url = f"https://github.com/{event.repository}"
    # Get the standard PR response template
    pr_response = f"""👋 Hello @{username}, thank you for submitting an `{event.repository}` 🚀 PR! To ensure a seamless integration of your work, please review the following checklist:

- ✅ **Define a Purpose**: Clearly explain the purpose of your fix or feature in your PR description, and link to any [relevant issues](https://github.com/{event.repository}/issues). Ensure your commit messages are clear, concise, and adhere to the project's conventions.
- ✅ **Synchronize with Source**: Confirm your PR is synchronized with the `{event.repository}` `main` branch. If it's behind, update it by clicking the 'Update branch' button or by running `git pull` and `git merge main` locally.
- ✅ **Ensure CI Checks Pass**: Verify all Ultralytics [Continuous Integration (CI)](https://docs.ultralytics.com/help/CI/) checks are passing. If any checks fail, please address the issues.
- ✅ **Update Documentation**: Update the relevant [documentation](https://docs.ultralytics.com/) for any new or modified features.
- ✅ **Add Tests**: If applicable, include or update tests to cover your changes, and confirm that all tests are passing.
- ✅ **Sign the CLA**: Please ensure you have signed our [Contributor License Agreement](https://docs.ultralytics.com/help/CLA/) if this is your first Ultralytics PR by writing "I have read the CLA Document and I sign the CLA" in a new message.
- ✅ **Minimize Changes**: Limit your changes to the **minimum** necessary for your bug fix or feature addition. _"It is not daily increase but daily decrease, hack away the unessential. The closer to the source, the less wastage there is."_  — Bruce Lee

For more guidance, please refer to our [Contributing Guide](https://docs.ultralytics.com/help/contributing/). Don't hesitate to leave a comment if you have any questions. Thank you for contributing to Ultralytics! 🚀"""

    example_pr_response = os.getenv("FIRST_PR_RESPONSE") or pr_response

    prompt = f"""Analyze this {event.repository} pull request and provide a comprehensive response.

SUMMARY INSTRUCTIONS:
- Generate a concise PR summary focusing on major changes, purpose, and impact for users
- Format with sections: ### 🌟 Summary, ### 📊 Key Changes, ### 🎯 Purpose & Impact

LABELS INSTRUCTIONS:
- Select most relevant labels from available options
- Only use "Alert" for obvious spam/abuse content

FIRST COMMENT INSTRUCTIONS:
- Start with the EXACT template provided below, including all badges, links and references
- KEEP ALL CHECKLIST ITEMS AND LINKS UNCHANGED from the example
- After the template, add a customized "PR-specific notes" section that addresses this specific PR:
  - Analyze the PR diff and title to identify key changes and potential concerns
  - Provide specific feedback on implementation approach, file changes, or testing needs
  - Highlight any backward compatibility, configuration, or setup considerations
  - Suggest specific improvements or verification steps relevant to this PR
  - Use emojis to make the notes engaging and scannable
- Format the PR-specific notes like:

PR-specific notes for "{title}":
- 🔧 [Specific technical feedback based on the actual changes]
- 📝 [Documentation or setup considerations]
- 🧪 [Testing recommendations]
- 🔄 [Configuration or compatibility notes]

AVAILABLE LABELS:
{chr(10).join(f"- {name}: {desc}" for name, desc in label_descriptions.items())}

EXAMPLE PR RESPONSE:
{example_pr_response}

REPOSITORY CONTEXT:
- Repository: {repo_name}
- Organization: {org_name} 
- Repository URL: {repo_url}
- Author: @{username}

PR TITLE:
{title}

PR DESCRIPTION:
{body[:2000] if body else "No description provided"}

PR DIFF:
{diff[:32000]}

Respond with JSON containing summary, labels array, and first_comment."""

    try:
        print("🤖 Making unified OpenAI API call...")
        response = get_completion(
            messages=[
                {
                    "role": "system",
                    "content": f"You are an Ultralytics AI assistant for GitHub PR analysis for {org_name}. Generate accurate, helpful responses for pull request management.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format=json_schema,
            check_links=False,  # Skip link checking for JSON responses
        )

        print("✅ Unified OpenAI call successful, parsing response...")
        data = json.loads(response)
        summary = SUMMARY_START + data.get("summary", "")
        labels = [label for label in data.get("labels", []) if label in label_descriptions]
        comment = data.get("first_comment", "")

        print(f"📋 Generated summary length: {len(summary)} chars")
        print(f"🏷️ Suggested labels: {labels}")
        print(f"💬 Comment length: {len(comment)} chars")
        print("✅ Unified PR analysis completed successfully")
        return summary, labels, comment

    except Exception as e:
        print(f"❌ Unified call failed ({e}), using individual functions")
        # Fallback to existing individual functions

        summary = generate_pr_summary(event.repository, diff)
        labels = get_relevant_labels(
            "pull request",
            title,
            body,
            label_descriptions,
            [],
        )
        comment = get_first_interaction_response(event, "pull request", title, body, username)
        return summary, labels, comment


def main(*args, **kwargs):
    """Summarize and label a PR and respond to the author."""
    event = Action(*args, **kwargs)
    action = event.event_data.get("action", "")

    print(f"Retrieving diff for PR {event.pr['number']}")
    print(f"Event action: {action}")  # DEBUG
    print(f"Event name: {event.event_name}")  # DEBUG

    # Unified approach for opened PRs (summary + labels + comment)
    print(f"Processing PR {event.pr['number']} with action: {action}")
    if action in {"opened", "reopened"}:
        summary, labels, first_comment = generate_unified_pr_response(event)

        # Update PR description
        print("Updating PR description...")
        update_pr_description(event, summary)

        # Apply labels if any were suggested
        if labels:
            print(f"Applying labels: {labels}")
            from .first_interaction import apply_labels

            apply_labels(event, event.pr["number"], event.pr.get("node_id"), labels, "pull request")

        # Add first comment
        if first_comment:
            print("Adding welcome comment...")
            add_comment(event, event.pr["number"], event.pr.get("node_id"), first_comment, "pull request")

    # Other actions
    elif action in ["synchronize", "edited"]:
        print("Updating PR summary...")
        diff = event.get_pr_diff()
        summary = generate_pr_summary(event.repository, diff)
        print("Updating PR description...")
        update_pr_description(event, summary)

    # Update linked issues and post thank you message if merged
    if event.pr.get("merged"):
        print("PR is merged, labeling fixed issues...")
        pr_credit = label_fixed_issues(
            event, summary if "summary" in locals() else generate_pr_summary(event.repository, event.get_pr_diff())
        )
        print("Removing TODO label from PR...")
        remove_pr_labels(event, labels=["TODO"])
        if pr_credit:
            print("Posting PR author thank you message...")
            post_merge_message(event, summary if "summary" in locals() else "", pr_credit)


if __name__ == "__main__":
    main()

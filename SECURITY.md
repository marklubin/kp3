# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

Please report security vulnerabilities by emailing mark@kairix.org.

**Do NOT create public GitHub issues for security vulnerabilities.**

We will:
- Respond within 48 hours to acknowledge your report
- Work with you to understand and validate the issue
- Keep you informed of our progress
- Credit you in the security advisory (unless you prefer to remain anonymous)

## Security Considerations

KP3 stores and processes text data with embeddings. When deploying KP3, consider:

- **Database Security**: Use strong credentials and network isolation for PostgreSQL
- **API Keys**: Protect your OpenAI/DeepSeek API keys; use environment variables, not code
- **Agent Isolation**: Use unique agent IDs to maintain data isolation between agents
- **Network Access**: The REST API should be deployed behind authentication in production

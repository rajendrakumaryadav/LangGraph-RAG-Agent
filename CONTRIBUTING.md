# Contributing to LangGraph RAG Agent 🤝

Thank you for your interest in contributing to the LangGraph RAG Agent project! This document provides guidelines and best practices for contributing code, documentation, and other improvements.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Review Process](#code-review-process)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Standards](#documentation-standards)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Issue Reporting](#issue-reporting)
- [Community](#community)

---

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors. We expect all participants to:

- ✅ Be respectful and constructive in all interactions
- ✅ Welcome newcomers and help them get started
- ✅ Accept constructive criticism gracefully
- ✅ Focus on what's best for the project and community
- ✅ Show empathy towards other community members

### Unacceptable Behavior

- ❌ Harassment, discrimination, or offensive comments
- ❌ Trolling or insulting/derogatory comments
- ❌ Personal or political attacks
- ❌ Publishing others' private information
- ❌ Other conduct that could be considered unprofessional

---

## 🚀 Getting Started

### Prerequisites

Before contributing, ensure you have:

1. **Python 3.12+** installed
2. **Git** for version control
3. **Docker & Docker Compose** (optional, for full stack development)
4. **UV package manager** (or pip)
5. **Text editor/IDE** (VS Code, PyCharm, etc.)

### Development Environment Setup

```bash
# 1. Fork the repository on GitHub
# Click "Fork" button on https://github.com/rajendrakumaryadav/LangGraph-RAG-Agent

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/LangGraph-RAG-Agent.git
cd LangGraph-RAG-Agent

# 3. Add upstream remote
git remote add upstream https://github.com/rajendrakumaryadav/LangGraph-RAG-Agent.git

# 4. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 5. Install dependencies
pip install uv
uv sync

# 6. Install development dependencies (if available)
pip install -e ".[dev]"  # Install in editable mode with dev dependencies

# 7. Start required services
docker-compose up -d

# 8. Pull required models
docker exec -it ollama ollama pull gemma3:1b
docker exec -it ollama ollama pull bge-m3

# 9. Configure environment
cp .env .env.local
# Edit .env.local with your settings
```

### Verify Your Setup

```bash
# Test that everything works
python -c "from core.rag import create_rag_workflow; print('✓ RAG workflow imports successfully')"

# Check code style
black --check .
isort --check-only .

# Run tests (if available)
pytest
```

---

## 🔄 Development Workflow

### Branch Strategy

We follow a simplified Git Flow model:

```
main (production-ready code)
  │
  ├── feature/your-feature-name
  ├── fix/bug-description
  ├── docs/documentation-update
  └── refactor/improvement-description
```

### Creating a New Branch

```bash
# Update your local main branch
git checkout main
git pull upstream main

# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### Branch Naming Conventions

- **Features**: `feature/add-csv-support`, `feature/user-authentication`
- **Bug Fixes**: `fix/retrieval-timeout`, `fix/memory-leak`
- **Documentation**: `docs/update-readme`, `docs/add-api-examples`
- **Refactoring**: `refactor/simplify-retriever`, `refactor/optimize-embeddings`
- **Tests**: `test/add-unit-tests`, `test/integration-tests`

### Making Changes

```bash
# Make your changes
# Edit files...

# Check status
git status

# Stage changes
git add .

# Commit with descriptive message
git commit -m "feat: add CSV file support for document ingestion"

# Push to your fork
git push origin feature/your-feature-name
```

---

## 🔍 Code Review Process

### Review Stages

All contributions go through a structured review process:

#### Stage 1: Automated Checks ⚙️

When you submit a PR, automated checks run:

- **Code Formatting**: Black, isort
- **Type Checking**: mypy (if configured)
- **Linting**: flake8, pylint
- **Tests**: pytest suite
- **Security**: Dependency vulnerability scanning

**Action Required**: Fix any failing automated checks before manual review.

#### Stage 2: Manual Code Review 👥

A maintainer will review your PR for:

1. **Code Quality**
   - Is the code clean, readable, and maintainable?
   - Does it follow project conventions?
   - Are variable names descriptive?
   - Is there adequate error handling?

2. **Functionality**
   - Does it solve the intended problem?
   - Are edge cases handled?
   - Does it integrate well with existing code?

3. **Testing**
   - Are there adequate tests?
   - Do tests cover edge cases?
   - Are tests clear and maintainable?

4. **Documentation**
   - Is new functionality documented?
   - Are docstrings clear and complete?
   - Is the README or other docs updated if needed?

5. **Performance**
   - Are there performance implications?
   - Is the implementation efficient?
   - Could it be optimized?

#### Stage 3: Feedback and Iteration 🔄

The reviewer may:

- **Approve**: Your PR is ready to merge! 🎉
- **Request Changes**: Address specific issues and re-submit
- **Comment**: Questions or suggestions for improvement
- **Reject**: Rare, but if the change doesn't align with project goals

**Responding to Feedback**:

```bash
# Make requested changes
# Edit files...

# Commit changes
git add .
git commit -m "refactor: address code review feedback"

# Push updates
git push origin feature/your-feature-name
```

The PR will automatically update with your new commits.

#### Stage 4: Merge ✅

Once approved:

1. Maintainer merges your PR
2. Your branch can be deleted
3. Your contribution is now part of the project!

```bash
# Clean up local branch (after merge)
git checkout main
git pull upstream main
git branch -d feature/your-feature-name
```

### Review Response Time

- **Simple fixes**: 1-2 days
- **Features**: 3-5 days
- **Major changes**: 1-2 weeks

If you haven't heard back within these timeframes, feel free to ping on the PR.

---

## 📝 Coding Standards

### Python Style Guide

We follow **PEP 8** with some project-specific conventions:

#### 1. Code Formatting

**Use Black** for automatic formatting:

```bash
# Format all Python files
black .

# Format specific file
black core/rag.py
```

**Configuration** (in `pyproject.toml`):
```toml
[tool.black]
line-length = 100
target-version = ['py312']
```

#### 2. Import Sorting

**Use isort** for consistent import ordering:

```bash
# Sort all imports
isort .

# Sort specific file
isort core/rag.py
```

**Import Order**:
```python
# 1. Standard library
import json
import sys
from typing import List, Dict

# 2. Third-party packages
from langchain_core.documents import Document
from langgraph.graph import StateGraph

# 3. Local modules
from core.models import get_generator_model
from core.retriever import create_retriever
```

#### 3. Type Hints

**Always use type hints** for function signatures:

```python
# ✅ Good
def retrieve_documents(question: str, k: int = 8) -> List[Document]:
    """Retrieve relevant documents."""
    pass

# ❌ Bad
def retrieve_documents(question, k=8):
    """Retrieve relevant documents."""
    pass
```

**For complex types**:
```python
from typing import List, Dict, Optional, Union, TypedDict

class GraphState(TypedDict):
    question: str
    documents: List[Document]
    generation: Optional[str]

def process_state(state: GraphState) -> Dict[str, str]:
    pass
```

#### 4. Docstrings

**Use Google-style docstrings**:

```python
def ingest_documents(paths: List[str], batch_size: int = 32) -> None:
    """
    Ingest documents from file paths into the vector database.

    This function loads documents, splits them into chunks, generates embeddings,
    and stores them in Qdrant for later retrieval.

    Args:
        paths: List of file paths to documents (PDF, DOCX, TXT)
        batch_size: Number of documents to process in each batch (default: 32)

    Raises:
        ValueError: If paths list is empty
        FileNotFoundError: If a specified file doesn't exist

    Example:
        >>> ingest_documents(['/path/to/doc.pdf'], batch_size=16)
        Ingested 45 chunks into Qdrant.
    """
    pass
```

#### 5. Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| **Functions** | snake_case | `retrieve_documents()` |
| **Variables** | snake_case | `final_answer` |
| **Classes** | PascalCase | `GraphState` |
| **Constants** | UPPER_SNAKE_CASE | `MAX_TOKENS` |
| **Private** | _leading_underscore | `_internal_helper()` |
| **Modules** | snake_case | `retriever.py` |

#### 6. Code Organization

```python
# Module-level constants
MAX_RETRIES = 3
DEFAULT_TIMEOUT = 30.0

# Module-level functions
def helper_function():
    """Helper that doesn't belong to a class."""
    pass

# Classes
class MyClass:
    """Class docstring."""
    
    def __init__(self):
        """Constructor."""
        pass
    
    def public_method(self):
        """Public method."""
        pass
    
    def _private_method(self):
        """Private helper method."""
        pass

# Main execution guard
if __name__ == "__main__":
    main()
```

#### 7. Error Handling

**Always handle expected errors gracefully**:

```python
# ✅ Good
try:
    documents = loader.load(path)
except FileNotFoundError:
    print(f"Error: File not found: {path}")
    return []
except Exception as e:
    print(f"Unexpected error loading {path}: {e}")
    return []

# ❌ Bad
try:
    documents = loader.load(path)
except:
    pass
```

#### 8. Logging

**Use print statements for user-facing messages, logging for debugging**:

```python
import logging

logger = logging.getLogger(__name__)

def process_documents(paths: List[str]) -> None:
    print(f"Processing {len(paths)} documents...")  # User-facing
    logger.debug(f"Paths: {paths}")  # Debug info
    
    for path in paths:
        logger.info(f"Loading {path}")  # Detailed progress
        # ... process
    
    print("✓ Processing complete!")  # User-facing
```

---

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── __init__.py
├── test_ingest.py          # Document ingestion tests
├── test_retriever.py       # Retrieval tests
├── test_rag.py             # RAG workflow tests
├── test_models.py          # Model tests
└── fixtures/               # Test data
    ├── sample.pdf
    ├── sample.txt
    └── mock_responses.json
```

### Writing Tests

#### Unit Tests

Test individual functions in isolation:

```python
import unittest
from unittest.mock import Mock, patch
from core.ingest import ingest_paths

class TestIngest(unittest.TestCase):
    """Test document ingestion functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_paths = ["tests/fixtures/sample.txt"]
    
    @patch('core.ingest.DoclingLoader')
    @patch('core.ingest.get_vector_store')
    def test_ingest_single_document(self, mock_store, mock_loader):
        """Test ingesting a single document."""
        # Setup mocks
        mock_loader.return_value.load.return_value = [
            Mock(page_content="Test content", metadata={})
        ]
        
        # Run test
        ingest_paths(self.test_paths)
        
        # Verify
        mock_loader.assert_called_once()
        mock_store.return_value.add_documents.assert_called_once()
    
    def test_ingest_empty_list(self):
        """Test that empty path list is handled gracefully."""
        # Should not raise an error
        ingest_paths([])
```

#### Integration Tests

Test multiple components working together:

```python
class TestRAGWorkflow(unittest.TestCase):
    """Test the complete RAG workflow."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once for all tests."""
        # Ingest test documents
        test_docs = ["tests/fixtures/sample.txt"]
        ingest_paths(test_docs)
    
    def test_end_to_end_workflow(self):
        """Test complete question-answer workflow."""
        from core.rag import create_rag_workflow
        
        app = create_rag_workflow()
        result = app.invoke({"question": "What is the test about?"})
        
        # Verify all stages completed
        self.assertIn("question", result)
        self.assertIn("documents", result)
        self.assertIn("generation", result)
        self.assertIn("final_answer", result)
        
        # Verify answer quality
        self.assertGreater(len(result["final_answer"]), 0)
        self.assertIn("**Sources:**", result["final_answer"])
```

### Test Coverage

Aim for **>80% code coverage** for core functionality:

```bash
# Install coverage tool
pip install coverage pytest-cov

# Run tests with coverage
pytest --cov=core --cov-report=html

# View coverage report
open htmlcov/index.html
```

### Test Naming

- Test files: `test_<module_name>.py`
- Test classes: `Test<Functionality>`
- Test methods: `test_<specific_behavior>`

**Examples**:
- `test_retrieve_with_empty_query()`
- `test_generate_answer_without_documents()`
- `test_critique_accepts_valid_answer()`

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_rag.py

# Run specific test
pytest tests/test_rag.py::TestRAGWorkflow::test_end_to_end_workflow

# Run with verbose output
pytest -v

# Run with output capture disabled (see prints)
pytest -s
```

---

## 📖 Documentation Standards

### When to Update Documentation

Update documentation when you:

- ✅ Add a new feature
- ✅ Change existing functionality
- ✅ Fix a bug that affects usage
- ✅ Add new configuration options
- ✅ Change API endpoints or interfaces
- ✅ Add new dependencies

### Documentation Files

| File | Purpose | Update When |
|------|---------|-------------|
| **README.md** | Main project documentation | Adding features, changing setup |
| **AGENTS.md** | Agent architecture details | Modifying agents, workflow |
| **CONTRIBUTING.md** | This file | Changing development process |
| **API.md** | API reference | Adding/changing API endpoints |
| **CHANGELOG.md** | Version history | Every release |

### Docstring Standards

Every public function/class should have a docstring:

```python
def create_retriever(vector_store: QdrantVectorStore, k: int = 8) -> BaseRetriever:
    """
    Create a multi-stage retriever for document search.

    This function creates a retriever that uses Maximum Marginal Relevance (MMR),
    Multi-Query generation, and Contextual Compression to find the most relevant
    documents for a given query.

    Args:
        vector_store: The Qdrant vector store containing document embeddings
        k: Number of documents to return (default: 8)

    Returns:
        A ContextualCompressionRetriever configured with MMR and Multi-Query

    Raises:
        ValueError: If k is less than 1

    Example:
        >>> from core.stores import get_vector_store
        >>> from core.models import get_embedding_model
        >>> embeddings = get_embedding_model()
        >>> store = get_vector_store(embeddings)
        >>> retriever = create_retriever(store, k=10)
        >>> docs = retriever.invoke("What is RAG?")

    Note:
        The retriever fetches 2.5x more documents (fetch_k) than it returns
        to allow for effective diversity filtering via MMR.
    """
    pass
```

### Code Comments

Use comments to explain **why**, not **what**:

```python
# ✅ Good - Explains reasoning
# Use MMR to balance relevance and diversity, preventing redundant results
retriever = vector_store.as_retriever(search_type="mmr")

# ❌ Bad - Obvious from code
# Create a retriever
retriever = vector_store.as_retriever(search_type="mmr")
```

**When to comment**:
- Complex algorithms or business logic
- Workarounds for external library issues
- Performance optimizations
- Security considerations
- Future TODOs (use `# TODO:` format)

```python
# TODO: Replace with async implementation when LangChain supports it
documents = retriever.invoke(question)

# HACK: Ollama sometimes returns malformed JSON, catch and retry
try:
    response = json.loads(llm_output)
except json.JSONDecodeError:
    response = _retry_with_fixed_format(llm_output)

# SECURITY: Sanitize user input to prevent injection attacks
safe_question = sanitize_input(question)
```

---

## 📬 Pull Request Guidelines

### Before Creating a PR

- [ ] Code follows project style guidelines
- [ ] All tests pass
- [ ] New tests added for new functionality
- [ ] Documentation updated
- [ ] Commit messages follow conventions
- [ ] Branch is up to date with main

```bash
# Update your branch with latest main
git checkout main
git pull upstream main
git checkout feature/your-feature
git rebase main
```

### PR Title Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

**Format**: `<type>(<scope>): <description>`

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, no logic change)
- `refactor`: Code refactoring (no feature change)
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
- `feat(retriever): add support for CSV files`
- `fix(rag): resolve timeout in critique agent`
- `docs(readme): update installation instructions`
- `refactor(ingest): simplify document loading logic`
- `perf(embeddings): implement caching for frequently asked questions`

### PR Description Template

```markdown
## Description
Brief description of what this PR does.

## Motivation
Why is this change needed? What problem does it solve?

## Changes
- List of changes made
- Another change
- One more change

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
How has this been tested?

- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review performed
- [ ] Code commented where necessary
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests added and passing
- [ ] Dependent changes merged

## Screenshots (if applicable)
Add screenshots for UI changes.

## Additional Context
Any additional information or context about the PR.
```

### PR Size Guidelines

Keep PRs **small and focused**:

- ✅ **Small**: < 200 lines changed
- ⚠️ **Medium**: 200-500 lines changed
- ❌ **Large**: > 500 lines changed (consider splitting)

**Tips for keeping PRs small**:
1. One feature/fix per PR
2. Break large features into smaller PRs
3. Submit documentation changes separately
4. Refactoring and features in separate PRs

### Draft PRs

Use **draft PRs** for work-in-progress:

```markdown
## [WIP] Add CSV file support

This is a work in progress. Feedback welcome on approach.

**TODO**:
- [ ] Implement CSV loader
- [ ] Add tests
- [ ] Update documentation
- [ ] Handle edge cases
```

---

## 🐛 Issue Reporting

### Before Creating an Issue

1. **Search existing issues** to avoid duplicates
2. **Check documentation** for known issues/solutions
3. **Verify you're using the latest version**
4. **Reproduce the issue** in a clean environment

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '...'
3. Run command '...'
4. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Error Messages**
```
Paste error messages or logs here
```

**Environment**
- OS: [e.g., Ubuntu 22.04, macOS 13.0]
- Python Version: [e.g., 3.12.1]
- Package Versions: [run `pip freeze` and paste relevant packages]
- Docker Version: [if using Docker]

**Additional Context**
Any other context about the problem.

**Possible Solution** (optional)
If you have ideas on how to fix it.
```

### Feature Request Template

```markdown
**Feature Description**
Clear description of the feature you'd like to see.

**Use Case**
Describe the use case and why this feature would be valuable.

**Proposed Solution**
How you envision this feature working.

**Alternatives Considered**
Alternative solutions or features you've considered.

**Additional Context**
Any other context, mockups, or examples.
```

### Issue Labels

Common labels used in this project:

| Label | Description |
|-------|-------------|
| `bug` | Something isn't working |
| `enhancement` | New feature or request |
| `documentation` | Documentation improvements |
| `good first issue` | Good for newcomers |
| `help wanted` | Extra attention needed |
| `question` | Further information requested |
| `wontfix` | This will not be worked on |
| `duplicate` | Duplicate of another issue |
| `priority: high` | High priority issue |
| `priority: low` | Low priority issue |

---

## 👥 Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Email**: rajendra.ecti@gmail.com

### Getting Help

- **Documentation**: Start with README.md and AGENTS.md
- **Examples**: Check `apps/` directory for usage examples
- **Issues**: Search existing issues for solutions
- **Ask**: Create a GitHub issue with your question

### Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in commit history

### Becoming a Maintainer

Regular contributors who demonstrate:
- Consistent high-quality contributions
- Active participation in code reviews
- Helping other community members
- Alignment with project values

May be invited to become maintainers with commit access.

---

## 📋 Review Checklist for Maintainers

When reviewing PRs, check:

### Code Quality
- [ ] Code follows project style (Black, isort)
- [ ] Type hints present and correct
- [ ] Error handling is appropriate
- [ ] No obvious bugs or issues
- [ ] Performance considerations addressed

### Testing
- [ ] Adequate test coverage (>80%)
- [ ] Tests are clear and maintainable
- [ ] Edge cases covered
- [ ] Tests pass in CI

### Documentation
- [ ] Docstrings updated for public APIs
- [ ] README or AGENTS.md updated if needed
- [ ] Comments explain complex logic
- [ ] Examples provided for new features

### Git Hygiene
- [ ] Commit messages follow conventions
- [ ] No merge commits (rebase instead)
- [ ] No unnecessary files committed
- [ ] Branch is up to date with main

### Architecture
- [ ] Changes fit project architecture
- [ ] No tight coupling introduced
- [ ] Modularity maintained
- [ ] No breaking changes (or documented)

### Security
- [ ] No secrets committed
- [ ] Input validation present
- [ ] No SQL injection vectors
- [ ] Dependencies are trusted

---

## 🎉 Thank You!

Thank you for contributing to LangGraph RAG Agent! Your efforts help make this project better for everyone.

**Remember**:
- Quality over quantity
- Be patient and kind
- Ask questions when unclear
- Have fun coding! 🚀

**Happy contributing! 🎊**

---

For more information:
- [README.md](README.md) - Project documentation
- [AGENTS.md](AGENTS.md) - Agent architecture guide
- [GitHub Repository](https://github.com/rajendrakumaryadav/LangGraph-RAG-Agent)

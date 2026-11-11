"""Generate domain-authentic AlPrime Capital finance sample documents (PDF, DOCX, TXT)."""

from pathlib import Path

from docx import Document as DocxDocument
from fpdf import FPDF

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "sample_docs"


def create_sample_pdf(output_path: Path) -> None:
    """Create a multi-page financial performance report PDF."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    pages = [
        (
            "ALPRIME CAPITAL CONSULTANCY\n"
            "Q3 2026 Financial Performance & Cost Analysis Report\n"
            "Location: Salhia Complex, Entrance 1, 4th Floor, Qibla, Kuwait City\n\n"
            "1. Executive Summary\n"
            "For the third quarter ended September 30, 2026, AlPrime Capital recorded total "
            "consolidated revenue of KWD 1,840,000, representing a 14.5% year-over-year increase "
            "compared to KWD 1,607,000 in Q3 2025. Gross profit reached KWD 1,196,000, lifting "
            "gross margin to 65.0%. Net operating income rose to KWD 478,400 (26.0% operating margin), "
            "driven by recurring CFO advisory retainers and automated cloud bookkeeping migrations.\n\n"
            "Key Operational Highlights:\n"
            "- Active corporate SME & family-office retainers grew from 68 to 84 clients.\n"
            "- Client retention rate for multi-year bookkeeping contracts stood at 94.2%.\n"
            "- Cash and short-term treasury reserves ended Q3 2026 at KWD 915,000 with zero bank debt."
        ),
        (
            "2. Segment Revenue & Operating Cost Breakdown (Q3 2026)\n\n"
            "Revenue by Practice Area:\n"
            "- Accounting & Cloud Bookkeeping: KWD 790,000 (42.9% of total revenue).\n"
            "- Fractional CFO & Financial Management: KWD 560,000 (30.4% of total revenue).\n"
            "- Cost Analysis & Internal Control Audits: KWD 295,000 (16.0% of total revenue).\n"
            "- Feasibility Studies & Market Research: KWD 195,000 (10.6% of total revenue).\n\n"
            "Operating Expense Breakdown:\n"
            "- Direct analyst and senior CPA payroll totaled KWD 812,000.\n"
            "- Technology, ERP licensing, and cloud security infrastructure cost KWD 138,000.\n"
            "- Salhia Complex office lease and general administration expenses totaled KWD 182,600.\n"
            "- Staff training and IFRS compliance certifications accounted for KWD 29,000."
        ),
        (
            "3. Liquidity, Receivables & Strategic Focus Notes\n\n"
            "Working Capital & Receivables:\n"
            "Average Days Sales Outstanding (DSO) improved from 46 days in Q2 2026 to 34 days in Q3 2026 "
            "after introducing automated milestone invoicing. Over 89% of accounts receivable are current "
            "within 30 days, and bad-debt provision remains capped at 0.8% of billed retainers.\n\n"
            "Capital Allocation & Hiring:\n"
            "Management approved KWD 110,000 for internal AI workflow tooling to speed up client financial "
            "statement reconciliation and regulatory memo drafting. Two senior IFRS specialists and one "
            "AI Data Analyst are scheduled to join the Kuwait City practice in November 2026.\n\n"
            "Disclaimer: Figures in this Q3 2026 internal report cover July 1 through September 30, 2026 "
            "in Kuwaiti Dinars (KWD). Full-year 2026 audited statements will be published in February 2027."
        ),
    ]

    for idx, content in enumerate(pages, start=1):
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 7, text=content)
        pdf.ln(4)
        pdf.set_font("Helvetica", style="I", size=9)
        pdf.cell(0, 6, text=f"AlPrime Capital - Confidential Q3 2026 Report (Page {idx})")

    pdf.output(str(output_path))


def create_sample_docx(output_path: Path) -> None:
    """Create an advisory fee schedule DOCX containing paragraphs and structured tables."""
    doc = DocxDocument()
    doc.add_heading("AlPrime Capital - 2026 Corporate Advisory & Retainer Fee Schedule", level=1)
    doc.add_paragraph(
        "Effective January 1, 2026, all corporate engagements in Kuwait follow the standardized "
        "monthly retainer tiers and fixed-scope project schedules outlined below."
    )

    doc.add_heading("1. Standard Monthly Retainer Packages", level=2)
    retainer_rows = [
        ["Service Tier", "Monthly Fee (KWD)", "Included Analyst Hours", "Primary Deliverable"],
        ["Essential Bookkeeping", "450 KWD", "20 hours", "Monthly Trial Balance & P&L"],
        ["Growth Financial Control", "950 KWD", "45 hours", "Monthly Close, Cash Flow & Cost Variance"],
        [
            "Fractional CFO Advisory",
            "1,850 KWD",
            "80 hours",
            "Board Reporting, Budgeting & Treasury Advisory",
        ],
    ]
    table = doc.add_table(rows=len(retainer_rows), cols=4)
    table.style = "Table Grid"
    for r_i, row in enumerate(retainer_rows):
        for c_i, val in enumerate(row):
            table.cell(r_i, c_i).text = val

    doc.add_heading("2. One-Time Fixed-Scope Projects", level=2)
    project_rows = [
        ["Engagement Type", "Fixed Fee Range (KWD)", "Typical Turnaround"],
        ["SME Market Feasibility Study", "2,500 - 4,200 KWD", "4 to 6 weeks"],
        ["Unit Cost & Pricing Restructuring", "1,600 - 2,800 KWD", "3 weeks"],
        ["Historical Ledger Cleanup (12 Months)", "1,200 KWD", "10 business days"],
    ]
    p_table = doc.add_table(rows=len(project_rows), cols=3)
    p_table.style = "Table Grid"
    for r_i, row in enumerate(project_rows):
        for c_i, val in enumerate(row):
            p_table.cell(r_i, c_i).text = val

    doc.add_heading("3. Billing, SLA & Termination Terms", level=2)
    doc.add_paragraph(
        "Invoices are issued on the 1st business day of each month and are payable net 15 calendar days "
        "via KNET corporate transfer or wire to AlPrime Capital's Kuwait City treasury account. "
        "Overtime advisory hours beyond the tier cap are billed at KWD 35 per hour only with written "
        "prior client email approval."
    )
    doc.add_paragraph(
        "Standard retainer agreements require a minimum 3-month commitment and may be terminated "
        "thereafter with 30 days' written notice. Emergency weekend audit support carries a 1.5x SLA surcharge."
    )
    doc.save(str(output_path))


def create_sample_txt(output_path: Path) -> None:
    """Create a plain-text internal compliance and onboarding policy document."""
    content = """ALPRIME CAPITAL - CLIENT ONBOARDING & AML/KYC COMPLIANCE POLICY (REV 2026.2)

1. PURPOSE & SCOPE
This policy governs mandatory Know Your Customer (KYC), Anti-Money Laundering (AML), and data
confidentiality controls for all new advisory, bookkeeping, and feasibility engagements handled
by AlPrime Capital staff in Kuwait.

2. MANDATORY CLIENT DOCUMENTATION
Before opening an ERP workspace or accessing bank feeds, engagement leads must collect:
- Valid Kuwait Ministry of Commerce & Industry (MOCI) Commercial License copy.
- Authorized Signatory Civil ID and passport verification record.
- Ultimate Beneficial Ownership (UBO) declaration for any shareholder owning 25% or more equity.
- Most recent signed fiscal year financial statements (if operational for more than 12 months).

3. RISK TIERS & ESCALATION SLA
- Standard Risk Clients: Onboarding review completed within 2 business days by the Engagement Manager.
- High-Risk Clients (cross-border cash-intensive entities or Politically Exposed Persons): Mandatory
  escalation to the designated Compliance Officer within 24 hours (1 business day), requiring written
  Managing Partner sign-off prior to engagement activation.

4. DATA RETENTION & AI PRIVACY RULES
Client financial ledgers must remain segregated per tenant and must never be used to train external
public AI models. All uploaded working papers and audit trails are archived in encrypted storage
for a statutory minimum retention period of 5 years under Kuwaiti commercial recordkeeping regulations.
"""
    output_path.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Extended "trap" corpus. These documents deliberately contain values that a
# careless RAG system confuses: a prior quarter with similar metrics, a
# restated figure, a forward target that is not an actual, a dated headcount,
# boundary values in an approval matrix, future prices that differ from the
# 2026 fee schedule, and a second retention period distinct from the 5-year
# statutory one.
# ---------------------------------------------------------------------------


def create_q2_management_accounts_pdf(output_path: Path) -> None:
    """Create a 4-page Q2 2026 management accounts PDF (prior-period trap document)."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pages = [
        (
            "ALPRIME CAPITAL CONSULTANCY\n"
            "Q2 2026 Management Accounts (Unaudited)\n"
            "Period: April 1 through June 30, 2026. All figures in Kuwaiti Dinars (KWD).\n\n"
            "1. Q2 2026 Headline Results\n"
            "Total consolidated revenue for Q2 2026 was KWD 1,712,000, up 3.4% from KWD 1,655,000 in Q1 2026. "
            "Gross profit was KWD 1,095,680, a gross margin of 64.0%. Net operating income was KWD 412,600, "
            "an operating margin of 24.1%.\n\n"
            "Active corporate SME & family-office retainers grew from 61 to 68 clients during the quarter. "
            "Cash and short-term treasury reserves stood at KWD 802,000 at June 30, 2026, with zero bank debt. "
            "Average Days Sales Outstanding (DSO) was 46 days."
        ),
        (
            "2. Q2 2026 Revenue by Practice Area\n\n"
            "- Accounting & Cloud Bookkeeping: KWD 741,000 (43.3% of total revenue).\n"
            "- Fractional CFO & Financial Management: KWD 498,000 (29.1% of total revenue).\n"
            "- Cost Analysis & Internal Control Audits: KWD 284,000 (16.6% of total revenue).\n"
            "- Feasibility Studies & Market Research: KWD 189,000 (11.0% of total revenue).\n\n"
            "First-half (H1) 2026 consolidated revenue was KWD 3,367,000 (Q1 2026 plus Q2 2026)."
        ),
        (
            "3. Restatement Note\n\n"
            "Q1 2026 revenue has been restated from the previously reported KWD 1,671,000 to KWD 1,655,000. "
            "The KWD 16,000 difference relates to pass-through software licence reimbursements that were "
            "reclassified from revenue to a reduction of technology expense. Q1 2026 net operating income is "
            "unchanged by the restatement.\n\n"
            "4. Headcount\n"
            "AlPrime Capital employed 42 full-time staff at June 30, 2026: 31 client-facing analysts and CPAs, "
            "6 technology and data staff, and 5 administration and compliance staff."
        ),
        (
            "5. Outlook & Targets (Forward-Looking, Not Actual Results)\n\n"
            "Management has set a Q4 2026 revenue target of KWD 1,950,000. Full-year 2026 revenue guidance is "
            "KWD 7,100,000 to KWD 7,250,000. These are internal targets only and are not reported results.\n\n"
            "Key risks to the outlook: delays in client ERP migrations, senior CPA hiring lead times of 10 to 14 "
            "weeks, and concentration risk, as the ten largest retainers represent 31% of Q2 2026 revenue."
        ),
    ]
    for idx, content in enumerate(pages, start=1):
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        pdf.multi_cell(0, 7, text=content)
        pdf.ln(4)
        pdf.set_font("Helvetica", style="I", size=9)
        pdf.cell(0, 6, text=f"AlPrime Capital - Q2 2026 Management Accounts (Page {idx})")
    pdf.output(str(output_path))


def _add_table(doc, rows) -> None:
    table = doc.add_table(rows=len(rows), cols=len(rows[0]))
    table.style = "Table Grid"
    for r_i, row in enumerate(rows):
        for c_i, val in enumerate(row):
            table.cell(r_i, c_i).text = val


def create_expense_policy_docx(output_path: Path) -> None:
    """Create a staff expense & approval policy DOCX (table + boundary-value trap document)."""
    doc = DocxDocument()
    doc.add_heading("AlPrime Capital - Staff Expense & Spend Approval Policy (v3.1)", level=1)
    doc.add_paragraph(
        "Effective March 1, 2026. Policy owner: Finance Director. Applies to all AlPrime Capital staff "
        "and contractors incurring business expenses on behalf of the firm."
    )

    doc.add_heading("1. Spend Approval Matrix", level=2)
    _add_table(
        doc,
        [
            ["Spend Amount (per transaction)", "Required Approver", "Approval Turnaround"],
            ["Up to and including KWD 500", "Engagement Manager", "1 business day"],
            ["KWD 501 to KWD 5,000", "Finance Director", "2 business days"],
            ["Above KWD 5,000", "Managing Partner", "5 business days"],
        ],
    )
    doc.add_paragraph(
        "Splitting a single purchase into smaller transactions to stay below an approval threshold is a "
        "policy breach and is reported to the Compliance Officer."
    )

    doc.add_heading("2. Travel Per Diem Rates", level=2)
    _add_table(
        doc,
        [
            ["Destination", "Daily Per Diem (KWD)", "Hotel Cap per Night (KWD)"],
            ["Within Kuwait (overnight only)", "15 KWD", "Not applicable"],
            ["GCC countries", "45 KWD", "60 KWD"],
            ["International (outside GCC)", "70 KWD", "95 KWD"],
        ],
    )
    doc.add_paragraph(
        "Economy class is mandatory for flights under 6 hours; business class may be booked for flights "
        "of 6 hours or more with Finance Director approval."
    )

    doc.add_heading("3. Claims, Receipts & Exclusions", level=2)
    doc.add_paragraph(
        "Expense claims must be submitted with itemised receipts within 10 business days of the expense "
        "date. Claims submitted after 30 calendar days are not reimbursed. Client entertainment is capped "
        "at KWD 25 per attendee. Alcohol, traffic fines, and personal entertainment are never reimbursable. "
        "Corporate cards carry a monthly limit of KWD 1,500 for managers and KWD 600 for analysts."
    )
    doc.save(str(output_path))


def create_engagement_faq_txt(output_path: Path) -> None:
    """Create a long client engagement FAQ TXT (temporal-conflict and scope trap document)."""
    content = """ALPRIME CAPITAL - CLIENT ENGAGEMENT FAQ (UPDATED OCTOBER 2026)

Q1. How long does onboarding take after we sign?
Engagement kickoff happens within 5 business days after KYC/AML clearance is complete. KYC/AML clearance
itself follows the timelines in the Client Onboarding & AML/KYC Compliance Policy.

Q2. Will retainer prices change in 2027?
Yes. Effective January 1, 2027, the Essential Bookkeeping tier increases to KWD 495 per month and the
Growth Financial Control tier increases to KWD 1,025 per month. The Fractional CFO Advisory tier remains
unchanged at KWD 1,850 per month. Included analyst hours do not change. Existing clients keep their 2026
price until their next contract renewal date.

Q3. Is there a discount for paying annually?
Clients who prepay 12 months in advance receive a 10% discount. The annual prepayment discount applies only
to the Growth Financial Control and Fractional CFO Advisory tiers. It does not apply to Essential
Bookkeeping or to one-time fixed-scope projects.

Q4. What are your support hours?
Client support is available Sunday to Thursday, 8:00 AM to 5:00 PM Kuwait time (GMT+3). Requests received
outside these hours are answered on the next business day, except emergency weekend audit support, which is
billed under the fee schedule surcharge.

Q5. In which languages are deliverables provided?
Board packs, management reports and feasibility studies are delivered in English and Arabic. Statutory
filings are prepared in Arabic.

Q6. How long do you keep our emails and system access logs?
Email correspondence and system access logs are retained for 2 years and then deleted. This is separate from
working papers and audit trails, which follow the 5-year statutory retention period in the compliance policy.

Q7. Which accounting systems do you support?
AlPrime Capital supports Zoho Books, Xero, QuickBooks Online, Odoo, and Microsoft Dynamics 365 Business
Central. Migrations from desktop or spreadsheet-based ledgers are quoted as a Historical Ledger Cleanup project.

Q8. Can we change tiers mid-contract?
Upgrades take effect from the next monthly invoice. Downgrades are allowed only after the minimum 3-month
commitment and require 30 days' written notice.
"""
    output_path.write_text(content, encoding="utf-8")


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    create_sample_pdf(SAMPLE_DIR / "AlPrime_Q3_2026_Financial_Performance.pdf")
    create_sample_docx(SAMPLE_DIR / "AlPrime_Advisory_Fee_Schedule_2026.docx")
    create_sample_txt(SAMPLE_DIR / "Kuwait_Market_Compliance_Policy.txt")
    create_q2_management_accounts_pdf(SAMPLE_DIR / "AlPrime_Q2_2026_Management_Accounts.pdf")
    create_expense_policy_docx(SAMPLE_DIR / "AlPrime_Staff_Expense_and_Approval_Policy.docx")
    create_engagement_faq_txt(SAMPLE_DIR / "AlPrime_Client_Engagement_FAQ.txt")
    print(f"Generated sample documents in: {SAMPLE_DIR}")


if __name__ == "__main__":
    main()

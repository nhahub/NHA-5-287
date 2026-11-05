"""Builds the PDF fixtures used by the PDF tests (needs reportlab, a test-only dependency)."""
import io


def table_pdf(n_rows: int = 70, header_every_page: bool = True) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
    rows = [["Region", "Units", "Revenue", "Margin"]] + [[f"R{i % 5}", str(10 + i), f"${1000 + i * 12:,}.50", f"{20 + i % 9}%"] for i in range(n_rows)]
    buf = io.BytesIO()
    t = Table(rows, repeatRows=1 if header_every_page else 0)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("FONTSIZE", (0, 0), (-1, -1), 10)]))
    SimpleDocTemplate(buf, pagesize=letter).build([t])
    return buf.getvalue()


def text_pdf(pages: int = 4) -> bytes:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for p in range(1, pages + 1):
        c.setFont("Helvetica", 9)
        c.drawString(72, 760, "ACME Corp Quarterly Report")                     # running header
        c.drawString(72, 30, f"Page {p} of {pages}")                            # running footer
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, 700, f"{p}. Section {p}")
        c.setFont("Helvetica", 11)
        y = 670
        lines = [("The customer reported on page %d that the delivery was late and the cus-" % p, 16), ("tomer service team resolved the complaint quickly.", 30),
                 ("A second paragraph follows here with unique words for page %d." % p, 16), ("It continues on a second line.", 16)]
        for line, step in lines:
            c.drawString(72, y, line)
            y -= step
        c.showPage()
    c.save()
    return buf.getvalue()


def mixed_pdf() -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    st = getSampleStyleSheet()
    rows = [["Product", "Price", "Stock"]] + [[f"Item {i}", f"{9.99 + i:.2f}", str(100 - i)] for i in range(12)]
    t = Table(rows)
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black)]))
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=letter).build([Paragraph("Price list overview", st["Title"]), Paragraph("This catalogue lists every product we sell and its current stock level for the season.", st["BodyText"]), Spacer(1, 12), t])
    return buf.getvalue()

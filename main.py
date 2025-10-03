import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
import warnings
from matplotlib.backends.backend_pdf import PdfPages
from fpdf import FPDF
import os
import tempfile

warnings.filterwarnings('ignore')

# Create directories if they don't exist
os.makedirs('./data/processed/', exist_ok=True)
os.makedirs('./reports/', exist_ok=True)

class FinancialDataProcessor:
    def __init__(self):
        self.train_news = None
        self.train_candles = None
        self.results = {}
        self.figures = []
        self.tables = []
        self.temp_files = []  # To track temporary files for cleanup

    def __del__(self):
        """Clean up temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except:
                pass

    def load_data(self):
        """Load and validate input data"""
        data_dir = './data/raw/participants/'
        try:
            self.train_news = pd.read_csv(data_dir + 'train_news.csv')
            self.train_candles = pd.read_csv(data_dir + 'train_candles.csv')
            self.results['data_loading'] = {
                'news_rows': self.train_news.shape[0],
                'news_cols': self.train_news.shape[1],
                'candles_rows': self.train_candles.shape[0],
                'candles_cols': self.train_candles.shape[1]
            }

            # Create data loading table
            loading_table = pd.DataFrame({
                'Dataset': ['News Data', 'Candle Data'],
                'Rows': [self.train_news.shape[0], self.train_candles.shape[0]],
                'Columns': [self.train_news.shape[1], self.train_candles.shape[1]]
            })
            self.tables.append(('Data Loading Summary', loading_table))

        except Exception as e:
            raise Exception(f"Data loading error: {e}")

    def remove_duplicates(self):
        """Remove duplicate candle records"""
        initial_count = len(self.train_candles)
        duplicates = self.train_candles[self.train_candles.duplicated(subset=['ticker', 'begin'], keep=False)]

        total_diff = 0
        duplicate_groups = 0
        if not duplicates.empty:
            grouped = duplicates.groupby(['ticker', 'begin'])
            for (ticker, begin), group in grouped:
                if len(group) > 1:
                    price_diff = group['close'].max() - group['close'].min()
                    total_diff += price_diff
                    duplicate_groups += 1

        self.train_candles = self.train_candles.drop_duplicates(subset=['ticker', 'begin'], keep='first')
        final_count = len(self.train_candles)

        self.results['duplicates'] = {
            'initial_count': initial_count,
            'final_count': final_count,
            'removed_duplicates': initial_count - final_count,
            'duplicate_groups': duplicate_groups,
            'total_price_diff': total_diff
        }

        # Create duplicates table
        duplicates_table = pd.DataFrame({
            'Metric': ['Initial Records', 'Final Records', 'Duplicates Removed', 'Duplicate Groups', 'Total Price Difference'],
            'Value': [initial_count, final_count, initial_count - final_count, duplicate_groups, f"{total_diff:.6f}"]
        })
        self.tables.append(('Duplicate Processing Results', duplicates_table))

        # Create visualization for duplicates
        fig, ax = plt.subplots(figsize=(10, 6))
        categories = ['Initial', 'Final', 'Removed']
        values = [initial_count, final_count, initial_count - final_count]
        bars = ax.bar(categories, values, color=['lightblue', 'lightgreen', 'lightcoral'])
        ax.set_title('Data Records: Before and After Duplicate Removal')
        ax.set_ylabel('Number of Records')
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100,
                   f'{value:,}', ha='center', va='bottom')
        plt.tight_layout()
        self.figures.append(('duplicate_removal', fig))
        plt.close()

    def analyze_gaps(self):
        """Analyze gaps in candle data"""
        self.train_candles['begin'] = pd.to_datetime(self.train_candles['begin'])
        self.train_candles.sort_values(by='begin', inplace=True)
        self.train_candles['diff'] = self.train_candles.groupby(by='ticker')['begin'].diff().dt.days.astype('Int64')
        self.train_candles['weekday'] = self.train_candles['begin'].dt.weekday

        diff_value_counts = self.train_candles['diff'].value_counts(dropna=False).reset_index()
        diff_value_counts.columns = ['diff', 'count']
        diff_value_counts_sorted = diff_value_counts.sort_values(by='diff')

        large_gaps = self.train_candles[self.train_candles['diff'].notna() & (self.train_candles['diff'] > 5)]

        self.results['gaps'] = {
            'gap_distribution': diff_value_counts_sorted,
            'large_gaps_count': len(large_gaps),
            'unique_tickers': len(self.train_candles['ticker'].unique())
        }

        # Create gap analysis table
        gap_table = diff_value_counts_sorted.copy()
        gap_table['diff'] = gap_table['diff'].apply(lambda x: 'First Record' if pd.isna(x) else f'{int(x)} days')
        gap_table.columns = ['Gap Duration', 'Frequency']
        self.tables.append(('Gap Distribution Analysis', gap_table))

        # Create gap visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(50, 6))

        # Gap distribution chart
        gap_data_filtered = diff_value_counts_sorted[diff_value_counts_sorted['diff'] <= 7]
        bars = ax1.bar(gap_data_filtered['diff'].astype(str), gap_data_filtered['count'],
                      color='orange', alpha=0.7)
        ax1.set_xlabel('Gap Duration (days)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Distribution of Data Gaps (1-7 days)')
        ax1.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars, gap_data_filtered['count']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f'{count}', ha='center', va='bottom')

        # Large gaps info
        large_gaps_info = pd.DataFrame({
            'Metric': ['Large Gaps (>5 days)', 'Unique Tickers', 'Total Records'],
            'Value': [len(large_gaps), len(self.train_candles['ticker'].unique()), len(self.train_candles)]
        })
        ax2.axis('off')
        table = ax2.table(cellText=large_gaps_info.values,
                         colLabels=large_gaps_info.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        ax2.set_title('Data Overview')

        plt.tight_layout()
        self.figures.append(('gap_analysis', fig))
        plt.close()

    def assign_tickers_to_news(self):
        """Assign stock tickers to news articles based on keywords"""
        ticker_keywords = {
            'AFLT': ['Aeroflot', 'AFLT'],
            'ALRS': ['Alrosa', 'ALRS'],
            'CHMF': ['Severstal', 'CHMF'],
            'GAZP': ['Gazprom', 'GAZP'],
            'GMKN': ['Norilsk Nickel', 'GMKN', 'Norilsk'],
            'LKOH': ['Lukoil', 'LKOH'],
            'MAGN': ['Magnitogorsk', 'MAGN'],
            'MGNT': ['Magnit', 'MGNT'],
            'MOEX': ['Moscow Exchange', 'MOEX'],
            'MTSS': ['MTS', 'MTSS'],
            'NVTK': ['Novatek', 'NVTK'],
            'PHOR': ['PhosAgro', 'PHOR'],
            'PLZL': ['Polyus', 'PLZL'],
            'ROSN': ['Rosneft', 'ROSN'],
            'RUAL': ['Rusal', 'RUAL'],
            'SBER': ['Sberbank', 'SBER'],
            'SIBN': ['Sibur', 'SIBN'],
            'T': ['Tatneft', 'TATN', 'T'],
            'VTBR': ['VTB', 'VTBR']
        }

        word_to_tickers = {
            'bank': ['SBER', 'VTBR'],
            'oil': ['GAZP', 'ROSN', 'T', 'LKOH'],
            'metal': ['GMKN', 'MAGN', 'CHMF', 'ALRS', 'PLZL', 'RUAL'],
            'exchange': ['MOEX'],
            'aviation': ['AFLT'],
            'telecom': ['MTSS'],
            'gold': ['PLZL'],
            'diamond': ['ALRS'],
            'gas': ['GAZP', 'NVTK'],
            'retail': ['MGNT'],
            'fertilizer': ['PHOR'],
        }

        def find_related_tickers(text: str):
            related = []
            text_lower = text.lower()

            for ticker, kw_list in ticker_keywords.items():
                ticker_mentioned_cnt = 0
                for kw in kw_list:
                    ticker_mentioned_cnt += text_lower.count(kw.lower())
                if ticker_mentioned_cnt > 0:
                    related.append((ticker, ticker_mentioned_cnt))

            related.sort(key=lambda x: x[1], reverse=True)
            related = [x[0] for x in related]

            for word, tickers in word_to_tickers.items():
                if word in text_lower:
                    for ticker in tickers:
                        if ticker not in related:
                            related.append(ticker)
            return ','.join(related)

        self.train_news['tickers'] = self.train_news.apply(
            lambda row: find_related_tickers(row['title'] + ' ' + row['publication']),
            axis=1
        )

        news_with_tickers = self.train_news[self.train_news['tickers'] != '']
        news_without_tickers = self.train_news[self.train_news['tickers'] == '']

        yandex_count = news_without_tickers[
            news_without_tickers['title'].str.lower().str.contains('yandex') |
            news_without_tickers['publication'].str.lower().str.contains('yandex')
        ].shape[0]

        self.results['ticker_assignment'] = {
            'news_with_tickers': len(news_with_tickers),
            'news_without_tickers': len(news_without_tickers),
            'yandex_mentions': yandex_count,
            'coverage_percentage': len(news_with_tickers) / len(self.train_news) * 100
        }

        # Create ticker assignment table
        assignment_table = pd.DataFrame({
            'Category': ['News with Tickers', 'News without Tickers', 'Yandex Mentions in Unassigned'],
            'Count': [len(news_with_tickers), len(news_without_tickers), yandex_count],
            'Percentage': [
                f"{len(news_with_tickers)/len(self.train_news)*100:.1f}%",
                f"{len(news_without_tickers)/len(self.train_news)*100:.1f}%",
                f"{yandex_count/len(news_without_tickers)*100:.1f}%" if len(news_without_tickers) > 0 else "0%"
            ]
        })
        self.tables.append(('Ticker Assignment Results', assignment_table))

        # Create assignment visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Pie chart
        labels = ['With Tickers', 'Without Tickers']
        sizes = [len(news_with_tickers), len(news_without_tickers)]
        colors = ['lightgreen', 'lightcoral']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Ticker Assignment Coverage')

        # Bar chart
        categories = ['Total News', 'Assigned', 'Unassigned', 'Yandex Mentions']
        values = [len(self.train_news), len(news_with_tickers), len(news_without_tickers), yandex_count]
        bars = ax2.bar(categories, values, color=['lightblue', 'lightgreen', 'lightcoral', 'orange'])
        ax2.set_title('Ticker Assignment Statistics')
        ax2.set_ylabel('Number of News Articles')
        ax2.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{value:,}', ha='center', va='bottom')

        plt.tight_layout()
        self.figures.append(('ticker_assignment', fig))
        plt.close()

        # Save processed data
        news_without_tickers.to_csv('./data/processed/empty_tickers.csv', index=False)
        news_with_tickers.to_csv('./data/processed/processed_tickers.csv', index=False)

    def analyze_ticker_mentions(self):
        """Analyze ticker mention statistics"""
        ticker_counts = self.train_news['tickers'].str.split(',', expand=True).stack().value_counts().reset_index()
        ticker_counts.columns = ['ticker', 'mentions']

        first_ticker = self.train_news['tickers'].str.split(',', n=1).str[0]
        first_ticker_counts = first_ticker.value_counts().reset_index()
        first_ticker_counts.columns = ['ticker', 'first_mentions']

        multiple_tickers = self.train_news[self.train_news['tickers'].str.contains(',')]
        ticker_counts_per_news = self.train_news['tickers'].str.split(',').apply(lambda x: len(x) if x != [''] else 0)

        self.results['ticker_analysis'] = {
            'ticker_counts': ticker_counts,
            'first_ticker_counts': first_ticker_counts,
            'multiple_tickers_count': len(multiple_tickers),
            'ticker_distribution': ticker_counts_per_news.value_counts().sort_index(),
            'avg_tickers_per_news': ticker_counts_per_news.mean()
        }

        # Create tickers table
        top_tickers_table = ticker_counts.copy()
        top_tickers_table['Percentage'] = (top_tickers_table['mentions'] / len(self.train_news) * 100).round(1)
        top_tickers_table.columns = ['Ticker', 'Mentions', 'Percentage (%)']
        self.tables.append(('Top Most Mentioned Tickers', top_tickers_table))

        # Create ticker distribution table
        dist_table = ticker_counts_per_news.value_counts().sort_index().reset_index()
        dist_table.columns = ['Tickers per News', 'Frequency']
        dist_table['Percentage'] = (dist_table['Frequency'] / len(self.train_news) * 100).round(1)
        self.tables.append(('Ticker Distribution per News Article', dist_table))

        # Create ticker analysis visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Top tickers horizontal bar chart
        top_15_tickers = ticker_counts
        bars = ax1.barh(top_15_tickers['ticker'], top_15_tickers['mentions'],
                       color='lightgreen', alpha=0.7)
        ax1.set_xlabel('Number of Mentions')
        ax1.set_title('Most Mentioned Tickers')
        ax1.invert_yaxis()
        for i, (ticker, mentions) in enumerate(zip(top_15_tickers['ticker'], top_15_tickers['mentions'])):
            ax1.text(mentions + 50, i, f'{mentions:,}', va='center')

        # Ticker distribution per news
        dist_data = ticker_counts_per_news.value_counts().sort_index()
        bars = ax2.bar(dist_data.index, dist_data.values, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Number of Tickers per News')
        ax2.set_ylabel('Number of News Articles')
        ax2.set_title('Distribution of Tickers per News Article')
        for bar, count in zip(bars, dist_data.values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{count}', ha='center', va='bottom')

        #  mentions
        first_15 = first_ticker_counts
        bars = ax3.bar(first_15['ticker'], first_15['first_mentions'], color='lightblue', alpha=0.7)
        ax3.set_xlabel('Ticker')
        ax3.set_ylabel('First Mentions')
        ax3.set_title('Ticker Mentions')
        ax3.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars, first_15['first_mentions']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                    f'{count}', ha='center', va='bottom')

        # Summary statistics
        stats_data = pd.DataFrame({
            'Metric': ['Multiple Ticker News', 'Average Tickers/News', 'Total Unique Tickers'],
            'Value': [f"{len(multiple_tickers):,}", f"{ticker_counts_per_news.mean():.2f}", f"{len(ticker_counts):,}"]
        })
        ax4.axis('off')
        table = ax4.table(cellText=stats_data.values,
                         colLabels=stats_data.columns,
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 2)
        ax4.set_title('Ticker Analysis Summary')

        plt.tight_layout()
        self.figures.append(('ticker_analysis', fig))
        plt.close()

    def analyze_gap_news(self):
        """Analyze news during gap periods"""
        gaps = self.train_candles[self.train_candles['diff'].notna() & (self.train_candles['diff'] > 5)]
        gap_info = []

        for idx, row in gaps.iterrows():
            ticker = row['ticker']
            gap_end = row['begin']
            gap_days = int(row['diff']) - 1
            gap_start = gap_end - timedelta(days=gap_days)
            missing_dates = [gap_start + timedelta(days=i) for i in range(1, gap_days+1)]

            for missing_date in missing_dates:
                news_count = self.train_news[
                    (self.train_news['tickers'].str.contains(ticker)) &
                    (pd.to_datetime(self.train_news['publish_date']).dt.date == missing_date.date())
                ].shape[0]
                gap_info.append({
                    'ticker': ticker,
                    'missing_date': missing_date.date(),
                    'news_count': news_count
                })

        gap_df = pd.DataFrame(gap_info)

        if not gap_df.empty:
            gap_stats = gap_df.groupby('ticker').agg({
                'missing_date': 'count',
                'news_count': 'sum'
            }).rename(columns={'missing_date': 'missed_days', 'news_count': 'total_news'})
            gap_stats['news_per_day'] = gap_stats['total_news'] / gap_stats['missed_days']
        else:
            gap_stats = pd.DataFrame()

        self.results['gap_analysis'] = {
            'gap_df': gap_df,
            'gap_stats': gap_stats,
            'total_gap_days': len(gap_df),
            'total_gap_news': gap_df['news_count'].sum() if not gap_df.empty else 0
        }

        # Create gap news table
        if not gap_df.empty:
            gap_summary_table = pd.DataFrame({
                'Metric': ['Total Gap Days', 'Total Gap News', 'Average News per Gap Day'],
                'Value': [
                    len(gap_df),
                    gap_df['news_count'].sum(),
                    f"{gap_df['news_count'].mean():.2f}"
                ]
            })
            self.tables.append(('Gap Period News Analysis', gap_summary_table))

            # Top gap news by ticker
            if not gap_stats.empty:
                top_gap_tickers = gap_stats.nlargest(50, 'total_news').reset_index()
                top_gap_tickers = top_gap_tickers[['ticker', 'missed_days', 'total_news', 'news_per_day']]
                top_gap_tickers.columns = ['Ticker', 'Missed Days', 'Total News', 'News per Day']
                self.tables.append(('Top 10 Tickers by Gap News Volume', top_gap_tickers))

            # Create gap news visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Gap news distribution
            if not gap_stats.empty:
                top_10_gap = gap_stats.nlargest(10, 'total_news')
                bars = ax1.bar(top_10_gap.index, top_10_gap['total_news'], color='purple', alpha=0.7)
                ax1.set_xlabel('Ticker')
                ax1.set_ylabel('Number of News Articles')
                ax1.set_title('Top 10 Tickers: News During Gap Periods')
                ax1.tick_params(axis='x', rotation=45)
                for bar, news_count in zip(bars, top_10_gap['total_news']):
                    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                            f'{int(news_count)}', ha='center', va='bottom')

            # Gap statistics
            gap_summary_data = pd.DataFrame({
                'Metric': ['Total Gap Days', 'Total Gap News', 'Avg News per Day'],
                'Value': [len(gap_df), gap_df['news_count'].sum(), f"{gap_df['news_count'].mean():.2f}"]
            })
            ax2.axis('off')
            table = ax2.table(cellText=gap_summary_data.values,
                             colLabels=gap_summary_data.columns,
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            table.scale(1, 2)
            ax2.set_title('Gap News Summary')

            plt.tight_layout()
            self.figures.append(('gap_news_analysis', fig))
            plt.close()

    def save_figure_to_temp(self, fig):
        """Save figure to temporary file and return filename"""
        temp_file = tempfile.mktemp(suffix='.png')
        fig.savefig(temp_file, dpi=150, bbox_inches='tight', format='png')
        self.temp_files.append(temp_file)
        return temp_file

    def generate_pdf_report(self):
        """Generate comprehensive PDF report with images and tables"""
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        pdf.add_page()
        pdf.set_font('Arial', 'B', 24)
        pdf.cell(0, 40, 'Financial News and Stock Price Analysis Report', 0, 1, 'C')
        pdf.set_font('Arial', '', 14)
        pdf.cell(0, 20, f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, 'C')
        pdf.ln(20)

        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, 'Executive Summary', 0, 1)
        pdf.set_font('Arial', '', 12)
        summary_text = f"""
This report presents a comprehensive analysis of financial news data and stock price data.
The analysis covers {self.results['data_loading']['news_rows']:,} news articles and {self.results['data_loading']['candles_rows']:,} price records
across {self.results['gaps']['unique_tickers']} stock tickers.

Key Findings:
- Ticker assignment coverage: {self.results['ticker_assignment']['coverage_percentage']:.1f}%
- Duplicate records removed: {self.results['duplicates']['removed_duplicates']}
- Large data gaps identified: {self.results['gaps']['large_gaps_count']}
- Average tickers per news: {self.results['ticker_analysis']['avg_tickers_per_news']:.2f}
- Total gap days with news: {self.results['gap_analysis']['total_gap_days']}
        """
        pdf.multi_cell(0, 8, summary_text)

        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, '1. Data Loading and Initial Assessment', 0, 1)

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '1.1 Data Loading Summary:', 0, 1)
        pdf.set_font('Arial', '', 10)

        loading_df = self.tables[0][1]
        col_widths = [60, 40, 40]

        pdf.set_fill_color(200, 200, 200)
        for i, col in enumerate(loading_df.columns):
            pdf.cell(col_widths[i], 10, str(col), 1, 0, 'C', True)
        pdf.ln()

        for _, row in loading_df.iterrows():
            for i, value in enumerate(row):
                pdf.cell(col_widths[i], 10, str(value), 1)
            pdf.ln()

        pdf.ln(10)

        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, '2. Data Quality: Duplicate Processing', 0, 1)

        for fig_name, fig in self.figures:
            if fig_name == 'duplicate_removal':
                temp_file = self.save_figure_to_temp(fig)
                pdf.image(temp_file, x=10, y=pdf.get_y(), w=180)
                pdf.ln(100)  # Space after image
                break

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '2.1 Duplicate Processing Results:', 0, 1)
        pdf.set_font('Arial', '', 10)

        duplicates_df = self.tables[1][1]
        col_widths = [80, 60]

        pdf.set_fill_color(200, 200, 200)
        for i, col in enumerate(duplicates_df.columns):
            pdf.cell(col_widths[i], 10, str(col), 1, 0, 'C', True)
        pdf.ln()

        for _, row in duplicates_df.iterrows():
            for i, value in enumerate(row):
                pdf.cell(col_widths[i], 10, str(value), 1)
            pdf.ln()

        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, '3. Data Continuity: Gap Analysis', 0, 1)

        for fig_name, fig in self.figures:
            if fig_name == 'gap_analysis':
                temp_file = self.save_figure_to_temp(fig)
                pdf.image(temp_file, x=10, y=pdf.get_y(), w=180)
                pdf.ln(100)
                break

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '3.1 Gap Distribution Analysis:', 0, 1)
        pdf.set_font('Arial', '', 10)

        gap_df = self.tables[2][1]
        col_widths = [60, 40]

        pdf.set_fill_color(200, 200, 200)
        for i, col in enumerate(gap_df.columns):
            pdf.cell(col_widths[i], 10, str(col), 1, 0, 'C', True)
        pdf.ln()

        for _, row in gap_df.iterrows():
            for i, value in enumerate(row):
                pdf.cell(col_widths[i], 10, str(value), 1)
            pdf.ln()

        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, '4. Ticker Assignment Analysis', 0, 1)

        for fig_name, fig in self.figures:
            if fig_name == 'ticker_assignment':
                temp_file = self.save_figure_to_temp(fig)
                pdf.image(temp_file, x=10, y=pdf.get_y(), w=180)
                pdf.ln(100)
                break

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '4.1 Ticker Assignment Results:', 0, 1)
        pdf.set_font('Arial', '', 10)

        assignment_df = self.tables[3][1]
        col_widths = [70, 40, 40]

        pdf.set_fill_color(200, 200, 200)
        for i, col in enumerate(assignment_df.columns):
            pdf.cell(col_widths[i], 10, str(col), 1, 0, 'C', True)
        pdf.ln()

        for _, row in assignment_df.iterrows():
            for i, value in enumerate(row):
                pdf.cell(col_widths[i], 10, str(value), 1)
            pdf.ln()

        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, '5. Detailed Ticker Analysis', 0, 1)

        for fig_name, fig in self.figures:
            if fig_name == 'ticker_analysis':
                temp_file = self.save_figure_to_temp(fig)
                pdf.image(temp_file, x=10, y=pdf.get_y(), w=180)
                pdf.ln(130)
                break

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '5.1 Top Most Mentioned Tickers:', 0, 1)
        pdf.set_font('Arial', '', 10)

        top_tickers_df = self.tables[4][1]
        col_widths = [40, 40, 40]

        pdf.set_fill_color(200, 200, 200)
        for i, col in enumerate(top_tickers_df.columns):
            pdf.cell(col_widths[i], 10, str(col), 1, 0, 'C', True)
        pdf.ln()

        for _, row in top_tickers_df.iterrows():
            for i, value in enumerate(row):
                pdf.cell(col_widths[i], 10, str(value), 1)
            pdf.ln()

        pdf.ln(5)

        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, '5.2 Ticker Distribution per News Article:', 0, 1)
        pdf.set_font('Arial', '', 10)

        dist_df = self.tables[5][1]
        col_widths = [50, 40, 40]

        pdf.set_fill_color(200, 200, 200)
        for i, col in enumerate(dist_df.columns):
            pdf.cell(col_widths[i], 10, str(col), 1, 0, 'C', True)
        pdf.ln()

        for _, row in dist_df.iterrows():
            for i, value in enumerate(row):
                pdf.cell(col_widths[i], 10, str(value), 1)
            pdf.ln()

        if self.results['gap_analysis']['total_gap_days'] > 0:
            pdf.add_page()
            pdf.set_font('Arial', 'B', 16)
            pdf.cell(0, 10, '6. Gap Period News Analysis', 0, 1)

            for fig_name, fig in self.figures:
                if fig_name == 'gap_news_analysis':
                    temp_file = self.save_figure_to_temp(fig)
                    pdf.image(temp_file, x=10, y=pdf.get_y(), w=180)
                    pdf.ln(150)
                    break

            gap_tables = [t for t in self.tables if 'Gap' in t[0]]
            for table_title, table_df in gap_tables:
                pdf.set_font('Arial', 'B', 12)
                pdf.cell(0, 10, f'6.1 {table_title}:', 0, 1)
                pdf.set_font('Arial', '', 10)

                col_widths = [len(str(col)) * 3 for col in table_df.columns]
                # Header
                pdf.set_fill_color(200, 200, 200)
                for i, col in enumerate(table_df.columns):
                    pdf.cell(col_widths[i], 10, str(col), 1, 0, 'C', True)
                pdf.ln()
                # Data
                for _, row in table_df.iterrows():
                    for i, value in enumerate(row):
                        pdf.cell(col_widths[i], 10, str(value), 1)
                    pdf.ln()
                pdf.ln(5)

        pdf.add_page()
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(0, 10, '7. Conclusions', 0, 1)
        pdf.set_font('Arial', '', 12)

        conclusions = [
            f"The dataset shows good quality with only {self.results['duplicates']['removed_duplicates']} duplicate records removed.",
            f"High success rate of {self.results['ticker_assignment']['coverage_percentage']:.1f}% in assigning tickers to news articles.",
            f"Oil and gas sector dominates news mentions.",
            f"Comprehensive coverage across {self.results['gaps']['unique_tickers']} tickers.",
            f"{self.results['ticker_analysis']['multiple_tickers_count']:,} news articles reference multiple tickers."
        ]

        for conclusion in conclusions:
            pdf.multi_cell(0, 8, f"- {conclusion}")

        # Save PDF
        pdf.output('./reports/financial_analysis_report.pdf')

        # Save visualizations to separate PDF
        with PdfPages('./reports/visualizations.pdf') as viz_pdf:
            for fig_name, fig in self.figures:
                viz_pdf.savefig(fig)

    def process_all(self):
        """Execute complete data processing pipeline"""
        print("Starting comprehensive financial data processing...")
        self.load_data()
        self.remove_duplicates()
        self.analyze_gaps()
        self.assign_tickers_to_news()
        self.analyze_ticker_mentions()
        self.analyze_gap_news()
        self.generate_pdf_report()

        self.train_candles.to_csv('./data/processed/cleaned_candles.csv', index=False)
        self.train_news.to_csv('./data/processed/cleaned_news.csv', index=False)

        print("\n" + "="*60)
        print("PROCESSING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("Generated Reports:")
        print("- ./reports/financial_analysis_report.pdf")
        print("- ./reports/visualizations.pdf")
        print("\nProcessed Data Files:")
        print("- ./data/processed/cleaned_candles.csv")
        print("- ./data/processed/cleaned_news.csv")
        print("- ./data/processed/empty_tickers.csv")
        print("- ./data/processed/processed_tickers.csv")
        print("="*60)

# Execute the processing
if __name__ == "__main__":
    processor = FinancialDataProcessor()
    processor.process_all()

-- Returns: product, revenue, sale_date.
select product, revenue, sale_date
from sales
order by revenue desc, sale_date asc
limit 3 offset 1
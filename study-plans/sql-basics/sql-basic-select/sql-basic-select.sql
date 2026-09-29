-- Returns: name, category, inventory_value.
select product_name as name, category as category, unit_price * units_in_stock as inventory_value
from products
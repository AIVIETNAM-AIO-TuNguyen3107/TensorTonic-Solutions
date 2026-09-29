-- Returns: customer_name, unique_products.
select * from (
    select customer_name, count(distinct(product)) as unique_products
    from orders
    group by customer_name
)
order by unique_products desc, customer_name asc nulls last
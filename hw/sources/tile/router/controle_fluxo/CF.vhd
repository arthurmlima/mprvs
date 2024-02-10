library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;




entity cf is
generic(
    x_init          : integer := 0;
    y_init          : integer := 0;
    img_width       : natural;
    img_height      : natural;
    n_frames        : natural;
    n_steps         : natural;
    pix_depth       : natural;
    subimg_width    : natural;
    subimg_height   : natural;
    buffer_length   : natural
);
    port (

  
	clk                             : in std_logic;
	reset                           : in std_logic;

	
    PM_in_pixel                     : in           std_logic_vector(pix_depth-1 downto 0);
    PM_in_x_dest                    : in           std_logic_vector(img_width-1 downto 0);
    PM_in_y_dest                    : in           std_logic_vector(img_height-1 downto 0);
    PM_in_step                      : in           std_logic_vector(n_steps-1 downto 0);
    PM_in_frame                     : in           std_logic_vector(n_frames-1 downto 0);
    PM_in_x_orig                    : in           std_logic_vector(img_width-1 downto 0);
    PM_in_y_orig                    : in           std_logic_vector(img_height-1 downto 0);
    PM_in_fb                        : in           std_logic;
    
    PM_stall_in_EB                  : inout        std_logic := '0'; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF 
    PM_stall_out_EB                 : in           std_logic := '0'; -- indica que o '''                                                                       
    PM_stall_in_DEC                 : inout        std_logic := '0'; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF 
    PM_stall_out_DEC                : in           std_logic := '0'; -- indica que o '''                        
    
    PM_in_direction                 : in           std_logic_vector(5 downto 0);
    

    PE_in_pixel                     : in           std_logic_vector(pix_depth-1 downto 0);
    PE_in_x_dest                    : in           std_logic_vector(img_width-1 downto 0);
    PE_in_y_dest                    : in           std_logic_vector(img_height-1 downto 0);
    PE_in_step                      : in           std_logic_vector(n_steps-1 downto 0);
    PE_in_frame                     : in           std_logic_vector(n_frames-1 downto 0);
    PE_in_x_orig                    : in           std_logic_vector(img_width-1 downto 0);
    PE_in_y_orig                    : in           std_logic_vector(img_height-1 downto 0);
    PE_in_fb                        : in           std_logic;

    PE_stall_in_EB                  : inout        std_logic := '0'; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF 
    PE_stall_out_EB                 : in           std_logic := '0'; -- indica que o '''                                                                       
    PE_stall_in_DEC                 : inout        std_logic := '0'; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF 
    PE_stall_out_DEC                : in           std_logic := '0'; 

    PE_in_direction                 : in           std_logic_vector(5 downto 0);
    





    N_in_pixel                      : in           std_logic_vector(pix_depth-1 downto 0);
    N_in_x_dest                     : in           std_logic_vector(img_width-1 downto 0);
    N_in_y_dest                     : in           std_logic_vector(img_height-1 downto 0);
    N_in_step                       : in           std_logic_vector(n_steps-1 downto 0);
    N_in_frame                      : in           std_logic_vector(n_frames-1 downto 0);
    N_in_x_orig                     : in           std_logic_vector(img_width-1 downto 0);
    N_in_y_orig                     : in           std_logic_vector(img_height-1 downto 0);
    N_in_fb                         : in           std_logic;
    
    N_stall_in_EB                   : inout        std_logic := '0'; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF 
    N_stall_out_EB                  : in           std_logic := '0'; -- indica que o '''                                                                       
    N_stall_in_DEC                  : inout        std_logic := '0'; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF 
    N_stall_out_DEC                 : in           std_logic := '0'; 
    

	
    N_in_direction              : in std_logic_vector(5 downto 0);
    





    S_in_pixel                  : in std_logic_vector(pix_depth-1 downto 0);
    S_in_x_dest                 : in std_logic_vector(img_width-1 downto 0);
    S_in_y_dest                 : in std_logic_vector(img_height-1 downto 0);
    S_in_step                   : in std_logic_vector(n_steps-1 downto 0);
    S_in_frame                  : in std_logic_vector(n_frames-1 downto 0);
    S_in_x_orig                 : in std_logic_vector(img_width-1 downto 0);
    S_in_y_orig                 : in std_logic_vector(img_height-1 downto 0);
    S_in_fb                     : in std_logic;
        
    S_stall_in_EB              : inout  std_logic := '0'; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF 
    S_stall_out_EB             : in     std_logic := '0'; -- indica que o '''                                                                       
    S_stall_in_DEC             : inout  std_logic := '0'; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF 
    S_stall_out_DEC            : in     std_logic := '0'; 
	
    
    S_in_direction              : in std_logic_vector(5 downto 0);

    E_in_pixel                  : in std_logic_vector(pix_depth-1 downto 0);
    E_in_x_dest                 : in std_logic_vector(img_width-1 downto 0);
    E_in_y_dest                 : in std_logic_vector(img_height-1 downto 0);
    E_in_step                   : in std_logic_vector(n_steps-1 downto 0);
    E_in_frame                  : in std_logic_vector(n_frames-1 downto 0);
    E_in_x_orig                 : in std_logic_vector(img_width-1 downto 0);
    E_in_y_orig                 : in std_logic_vector(img_height-1 downto 0);
    E_in_fb                     : in std_logic;
   
    E_stall_in_EB              : inout  std_logic := '0'; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF 
    E_stall_out_EB             : in     std_logic := '0'; -- indica que o '''                                                                       
    E_stall_in_DEC             : inout  std_logic := '0'; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF 
    E_stall_out_DEC            : in     std_logic := '0'; 

    E_in_direction              : in std_logic_vector(5 downto 0);
 

    W_in_pixel                  : in std_logic_vector(pix_depth-1 downto 0);
    W_in_x_dest                 : in std_logic_vector(img_width-1 downto 0);
    W_in_y_dest                 : in std_logic_vector(img_height-1 downto 0);
    W_in_step                   : in std_logic_vector(n_steps-1 downto 0);
    W_in_frame                  : in std_logic_vector(n_frames-1 downto 0);
    W_in_x_orig                 : in std_logic_vector(img_width-1 downto 0);
    W_in_y_orig                 : in std_logic_vector(img_height-1 downto 0);
    W_in_fb                     : in std_logic;
    
    W_stall_in_EB               : inout    std_logic := '0'; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF 
    W_stall_out_EB              : in     std_logic := '0'; -- indica que o '''                                                                       
    W_stall_in_DEC              : inout    std_logic := '0'; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF 
    W_stall_out_DEC             : in     std_logic := '0'; 
 
    W_in_direction              : in std_logic_vector(5 downto 0);
 
 
    
    
    i_PM_pixel          : out std_logic_vector(pix_depth-1 downto 0);
    i_PM_x_dest         : out std_logic_vector(img_width-1 downto 0);
    i_PM_y_dest         : out std_logic_vector(img_height-1 downto 0);
    i_PM_step           : out std_logic_vector(n_steps-1 downto 0);
    i_PM_frame          : out std_logic_vector(n_frames-1 downto 0);
    i_PM_x_orig         : out std_logic_vector(img_width-1 downto 0);
    i_PM_y_orig         : out std_logic_vector(img_height-1 downto 0);
    i_PM_fb             : out std_logic; -- identify if the it is a set_pixel or a set_pixel message.
    i_PM_busy           : in std_logic; -- se '1' ent?o t? livre 
    
    i_PM_req            : inout std_logic := '0'; 
    i_PM_ack            : in std_logic := '0'; 
    
   
    i_PE_pixel          : out std_logic_vector(pix_depth-1 downto 0);
    i_PE_x_dest         : out std_logic_vector(img_width-1 downto 0);
    i_PE_y_dest         : out std_logic_vector(img_height-1 downto 0);
    i_PE_step           : out std_logic_vector(n_steps-1 downto 0);
    i_PE_frame          : out std_logic_vector(n_frames-1 downto 0);
    i_PE_x_orig         : out std_logic_vector(img_width-1 downto 0);
    i_PE_y_orig         : out std_logic_vector(img_height-1 downto 0);
    i_PE_fb             : out std_logic; -- identify if the it is a set_pixel or a set_pixel message. 
    i_PE_busy           : in std_logic;
   
    i_PE_req            : inout std_logic:= '0'; 
    i_PE_ack            : in std_logic:= '0'; 




    i_N_pixel           : out std_logic_vector(pix_depth-1 downto 0);
    i_N_x_dest          : out std_logic_vector(img_width-1 downto 0);
    i_N_y_dest          : out std_logic_vector(img_height-1 downto 0);
    i_N_step            : out std_logic_vector(n_steps-1 downto 0);
    i_N_frame           : out std_logic_vector(n_frames-1 downto 0);
    i_N_x_orig          : out std_logic_vector(img_width-1 downto 0);
    i_N_y_orig          : out std_logic_vector(img_height-1 downto 0);
    i_N_fb              : out std_logic; -- message forward/backward 
    --i_N_busy            : in std_logic;
    
    i_N_req             : inout std_logic := '0'; 
    i_N_ack             : in std_logic := '0'; 
    



    i_S_pixel           : out std_logic_vector(pix_depth-1 downto 0);
    i_S_x_dest          : out std_logic_vector(img_width-1 downto 0);
    i_S_y_dest          : out std_logic_vector(img_height-1 downto 0);
    i_S_step            : out std_logic_vector(n_steps-1 downto 0);
    i_S_frame           : out std_logic_vector(n_frames-1 downto 0);
    i_S_x_orig          : out std_logic_vector(img_width-1 downto 0);
    i_S_y_orig          : out std_logic_vector(img_height-1 downto 0);
    i_S_fb              : out std_logic; -- message forward/backward  
  --  i_S_busy            : in std_logic;
    
    i_S_req            : inout std_logic := '0'; 
    i_S_ack            : in std_logic := '0'; 
    
    

    i_E_pixel           : out std_logic_vector(pix_depth-1 downto 0);
    i_E_x_dest          : out std_logic_vector(img_width-1 downto 0);
    i_E_y_dest          : out std_logic_vector(img_height-1 downto 0);
    i_E_step            : out std_logic_vector(n_steps-1 downto 0);
    i_E_frame           : out std_logic_vector(n_frames-1 downto 0);
    i_E_x_orig          : out std_logic_vector(img_width-1 downto 0);
    i_E_y_orig          : out std_logic_vector(img_height-1 downto 0);
    i_E_fb              : out std_logic; -- message forward/backward
  --  i_E_busy            : in std_logic;
    
    i_E_req            : inout std_logic := '0'; 
    i_E_ack            : in std_logic := '0'; 
      
    -- West
    i_W_pixel           : out std_logic_vector(pix_depth-1 downto 0);
    i_W_x_dest          : out std_logic_vector(img_width-1 downto 0);
    i_W_y_dest          : out std_logic_vector(img_height-1 downto 0);
    i_W_step            : out std_logic_vector(n_steps-1 downto 0);
    i_W_frame           : out std_logic_vector(n_frames-1 downto 0);
    i_W_x_orig          : out std_logic_vector(img_width-1 downto 0);
    i_W_y_orig          : out std_logic_vector(img_height-1 downto 0);
    i_W_fb              : out std_logic; -- message forward/backward   
   -- i_W_busy            : in std_logic;
    
    i_W_req            : inout std_logic := '0';  
    i_W_ack            : in std_logic := '0' 
   
    
    );
end entity cf;
 
architecture rtl of cf is



type bf_dir is
record
    in_ack      : std_logic; -- not part of the message, but flags if this position is clean or not.
    counter     : integer  range 0 to 5 ; -- signal from the decoder informing there is a new message to be sent
    direction : std_logic_vector(5 downto 0);
   -- channel : integer range 0 to 6;    
end record;


type bf_proc is
record
    
    vec      : std_logic;
    in_ack     : std_logic ; 
    direction : std_logic_vector(5 downto 0);
    channel : integer range 0 to 5;
    
   -- channel : integer range 0 to 6;    
end record;





type in_bf_dir is array(0 to (5)) of bf_dir;
type in_bf_proc is array(0 to (5)) of bf_proc;

type state_type is (xxx, yyy, zzz);  -- Define the states
signal contador_two : state_type := xxx;

-----------------------------------------
-----------------------------------------
-----------------------------------------

signal buffer_direction : in_bf_dir;
signal buffer_PROCEDURE : in_bf_proc;

SIGNAL CONTADOR : STD_LOGIC      := '0';
--SIGNAL CONTADOR_two : STD_LOGIC      := '0';

signal PM_SET_ultimo : std_logic := '0';
signal PE_SET_ultimo : std_logic := '0';
signal N_SET_ultimo : std_logic  := '0' ;
signal S_SET_ultimo : std_logic  := '0'; 
signal E_SET_ultimo : std_logic  := '0'; 
signal W_SET_ultimo : std_logic  := '0' ;

signal s_PM_busy : std_logic := '0';
signal s_PE_busy : std_logic := '0';
signal s_N_busy  : std_logic  := '0' ;
signal s_S_busy  : std_logic  := '0'; 
signal s_E_busy  : std_logic  := '0'; 
signal s_W_busy  : std_logic  := '0' ;
signal contador_mas : std_logic_vector(2 downto 0) := "000";

signal PM : std_logic;
signal PE : std_logic;
signal N : std_logic;
signal S : std_logic;
signal E : std_logic;
signal W : std_logic;

signal counter_PM : integer  range 0 to 5 := 0; 
signal counter_PE : integer range 0 to 5 := 1;
signal counter_N  : integer range 0 to 5 := 2;
signal counter_S  : integer range 0 to 5 := 3;
signal counter_E  : integer range 0 to 5 := 4;
signal counter_W  : integer range 0 to 5 := 5;
signal clk_count  : integer := 0;
signal alight     : std_logic  := '0';

signal s_PM_stall : std_logic ;
signal s_PE_stall : std_logic ;
signal s_N_stall : std_logic ;
signal s_S_stall : std_logic ;
signal s_E_stall : std_logic ;
signal s_W_stall : std_logic ;


--signal direction : std_logic_vector 5 downto 0 := "000000";
---------------------------------------

---------------------------------------




function arb(buffer_direction:in_bf_dir) return in_bf_proc is

    variable counter_l : integer range 0 to 5 := 0;
    variable counter_f : integer range 0 to 5 := 0;
    variable counter_q : integer range 0 to 5 := 0;
    variable max : integer range 0 to 5 := 0;
    variable j_max : integer range 0 to 5 := 0;
   
    
    constant  clean : bf_proc := (  vec     => '0', 
                                    in_ack  => '0',   
                                    direction => (others =>'0'),
                                    channel => 0);
                                    
    
    
    variable buffer_procedure_v : in_bf_proc := (others => clean);
    variable buffer_aux : in_bf_proc := (others => clean);

    begin
        --scan flags_v until finds a match
        
--        for counter_q in 0 to 5 loop 
        
--        buffer_aux(counter_q)
        
--        end loop;
        
        

        
       -- buffer_procedure_v(j_max).vec :='1';
               
        
        max := 0;
        j_maX := 0;
        for counter_f in 0 to 5 loop

        if (buffer_direction(counter_f).in_ack = '1') then 
        if (buffer_direction(counter_f).counter >= max) then
         max := buffer_direction(counter_f).counter;
         j_max := counter_f;
                
                buffer_procedure_v(0).channel :=  counter_f ;
                buffer_procedure_v(0).direction := buffer_direction(counter_f).direction;
                --buffer_procedure_v(counter_l).in_ack := '1';
                
                            
        end if;

        end if;

        end loop;

    return buffer_procedure_v;
end function arb;




begin


process(PM_in_direction,PE_in_direction,N_in_direction,S_in_direction,E_in_direction,W_in_direction,clk) is 
begin 
buffer_direction(0).direction <=  PM_in_direction; 
buffer_direction(1).direction <=  PE_in_direction; 
buffer_direction(2).direction <=  N_in_direction; 
buffer_direction(3).direction <=  S_in_direction; 
buffer_direction(4).direction <=  E_in_direction; 
buffer_direction(5).direction <=  W_in_direction; 
buffer_direction(0).in_ack <= PM_stall_out_EB and PM_stall_out_DEC  and s_PM_busy ;
buffer_direction(1).in_ack <= PE_stall_out_EB and PE_stall_out_DEC  and s_PE_busy ;
buffer_direction(2).in_ack <= N_stall_out_EB and N_stall_out_DEC  and s_N_busy    ;
buffer_direction(3).in_ack <= S_stall_out_EB and S_stall_out_DEC  and s_S_busy    ;
buffer_direction(4).in_ack <= E_stall_out_EB and E_stall_out_DEC  and s_E_busy    ;
buffer_direction(5).in_ack <= W_stall_out_EB and W_stall_out_DEC  and s_W_busy    ;


   
    if ( PM_in_direction = "010000" ) then
    s_PM_busy <= i_PE_busy;
    elsif (PM_in_direction = "100000") then 
    s_PM_busy <= i_PM_busy;
    else 
    s_PM_busy <= '1';
    end if;
    

    
    
    if ( PE_in_direction = "010000" ) then
    s_PE_busy <= i_PE_busy;
    elsif (PE_in_direction = "100000") then 
    s_PE_busy <= i_PM_busy;
    else 
    s_PE_busy <= '1';
    end if;
    
    if ( N_in_direction = "010000" ) then
    s_N_busy <= i_PE_busy;
    elsif (N_in_direction = "100000") then 
    s_N_busy <= i_PM_busy;
    else 
    s_N_busy <= '1';
    end if;
    
    if (S_in_direction = "010000" ) then
    s_S_busy <= i_PE_busy;
    elsif (S_in_direction = "100000") then 
    s_S_busy <= i_PM_busy;
    else 
    s_S_busy <= '1';
    end if;
    
    
     if ( E_in_direction = "010000" ) then
    s_E_busy <= i_PE_busy;
    elsif (E_in_direction = "100000") then 
    s_E_busy <= i_PM_busy;
    else 
    s_E_busy <= '1';
    end if;

    if ( W_in_direction = "010000" ) then
    s_W_busy <= i_PE_busy;
    elsif (W_in_direction = "100000") then 
    s_W_busy <= i_PM_busy;
    else 
    s_W_busy <= '1';
    end if;
    
buffer_procedure <= arb(buffer_direction); 






end process;











process (clk,reset,buffer_direction) is

begin

if ( reset = '1') then

i_PM_pixel  <= (others => '0');
i_PM_x_dest <= (others => '0');
i_PM_y_dest <= (others => '0');
i_PM_step   <= (others => '0');
i_PM_frame  <= (others => '0');
i_PM_x_orig <= (others => '0');
i_PM_y_orig <= (others => '0');
i_PM_fb     <= '0';


i_PE_pixel  <= (others => '0');
i_PE_x_dest <= (others => '0');
i_PE_y_dest <= (others => '0');
i_PE_step   <= (others => '0');
i_PE_frame  <= (others => '0');
i_PE_x_orig <= (others => '0');
i_PE_y_orig <= (others => '0');
i_PE_fb     <= '0';


i_N_pixel  <= (others => '0');
i_N_x_dest <= (others => '0');
i_N_y_dest <= (others => '0');
i_N_step   <= (others => '0');
i_N_frame  <= (others => '0');
i_N_x_orig <= (others => '0');
i_N_y_orig <= (others => '0');
i_N_fb     <= '0';


i_S_pixel  <= (others => '0');
i_S_x_dest <= (others => '0');
i_S_y_dest <= (others => '0');
i_S_step   <= (others => '0');
i_S_frame  <= (others => '0');
i_S_x_orig <= (others => '0');
i_S_y_orig <= (others => '0');
i_S_fb     <= '0';


i_E_pixel  <= (others => '0');
i_E_x_dest <= (others => '0');
i_E_y_dest <= (others => '0');
i_E_step   <= (others => '0');
i_E_frame  <= (others => '0');
i_E_x_orig <= (others => '0');
i_E_y_orig <= (others => '0');
i_E_fb     <= '0';



i_W_pixel  <= (others => '0');
i_W_x_dest <= (others => '0');
i_W_y_dest <= (others => '0');
i_W_step   <= (others => '0');
i_W_frame  <= (others => '0');
i_W_x_orig <= (others => '0');
i_W_y_orig <= (others => '0');
i_W_fb     <= '0';



PM_SET_ultimo <= '0';
PE_SET_ultimo <= '0';
N_SET_ultimo  <= '0';
S_SET_ultimo  <= '0';
E_SET_ultimo  <= '0';
W_SET_ultimo  <= '0';
PM_stall_in_EB <=  '0'; 
PM_stall_in_DEC <= '0';
PE_stall_in_EB <=  '0'; 
PE_stall_in_DEC <= '0';
N_stall_in_EB <=   '0'; 
N_stall_in_DEC <=  '0';
S_stall_in_EB <=   '0'; 
S_stall_in_DEC <=  '0';
E_stall_in_EB <=   '0'; 
E_stall_in_DEC <=  '0';
W_stall_in_EB <=   '0'; 
W_stall_in_DEC <=  '0';



elsif rising_edge(clk) then  
    
case contador is 
when '0' =>
--PM_SET_ultimo <= '0';
--PE_SET_ultimo <= '0';
--N_SET_ultimo  <= '0';
--S_SET_ultimo  <= '0';
--E_SET_ultimo  <= '0';
--W_SET_ultimo  <= '0';


--    if (PM_stall_out_DEC = '0') then 
    
--    s_PM_stall
    
    
--    end if;
--        if (E_stall_out_DEC = '0') then 
    
--    end if;
--        if (E_stall_out_DEC = '0') then 
    
--    end if;
    
--            if (E_stall_out_DEC = '0') then 
    
--    end if;
    
--            if (E_stall_out_DEC = '0') then 
    
--    end if;
    
--            if (E_stall_out_DEC = '0') then 
    
--    end if;
    
--            if (E_stall_out_DEC = '0') then 
    
--    end if;
    
    


 

  --  buffer_direction(0).in_ack <=  PM_stall_out_EB and PM_stall_out_DEC and s_PM_busy and PM_stall_out_DEC  ;
    
    if ( counter_PM <5 ) then
    counter_PM <=  counter_PM+1;
    end if;
    
    if(PM_SET_ultimo = '1') then
    PM_SET_ultimo <= '0';   
    counter_PM<= 0;
    end if;
    
  --  buffer_direction(0).counter <=  counter_PM;
   -- buffer_direction(0).direction <=  PM_in_direction;        
--    --------------------------------------------------------------
    
   -- buffer_direction(1).in_ack <=   PE_stall_out_EB and PE_stall_out_DEC and s_PE_busy and PE_stall_out_DEC ;   

    if ( counter_PE <5 ) then
    counter_PE <=  counter_PE+1;
    end if;
    
    
    if(PE_SET_ultimo = '1') then
    PE_SET_ultimo <= '0';   
    counter_PE<= 0;
    end if;
 --   buffer_direction(1).counter <=  counter_PE;
         
--    --------------------------------------------------------------
  

  --  buffer_direction(2).in_ack <= N_stall_out_EB and N_stall_out_DEC  and s_N_busy and N_stall_out_DEC;
 
    if ( counter_N <5 ) then
    counter_N <=  counter_N+1;

    end if;
    
    if(N_SET_ultimo = '1') then
    N_SET_ultimo <= '0';   
    counter_N<= 0;
    end if;
    
   -- buffer_direction(2).counter <=  counter_N;
    
  --  buffer_direction(2).direction <=  N_in_direction;        
--    --------------------------------------------------------------

 
  --  buffer_direction(3).in_ack <=   S_stall_out_EB and S_stall_out_DEC and s_S_busy and S_stall_out_DEC ;
  
    if ( counter_S <5 ) then
    counter_S <=  counter_S+1;
    end if;
    
    if(S_SET_ultimo = '1') then
    S_SET_ultimo <= '0';   
    counter_S<= 0;
    end if;
    

--    buffer_direction(3).counter <=  counter_S;
    
--    buffer_direction(3).direction <=  S_in_direction;        
    
----------------------------------------------------------------
----------------------------------------------------------------
    
    --buffer_direction(4).in_ack <=  E_stall_out_EB and E_stall_out_DEC and s_E_busy and E_stall_out_DEC  ;

    if ( counter_E <5 ) then
    counter_E <=  counter_E+1;

    end if;
    
    
    if(E_SET_ultimo = '1') then
    E_SET_ultimo <= '0';   
    counter_E<= 0;
    end if;
    
    

--    buffer_direction(4).counter <=  counter_E;
   
--    buffer_direction(4).direction <=  E_in_direction;        
    
-- --------------------------------------------------------------
--    --------------------------------------------------------------
   -- buffer_direction(5).in_ack <= W_stall_out_EB and W_stall_out_DEC and s_W_busy and W_stall_out_DEC  ;

    
    if ( counter_W <5 ) then
    counter_W <=  counter_W+1;

    end if;
    
    if(W_SET_ultimo = '1') then
    W_SET_ultimo <= '0';   
    counter_W<= 0;
    end if;
    
--    buffer_direction(5).counter <=  counter_W;  
--    buffer_direction(5).direction <=  W_in_direction;  
    
    
    
        
--    if ( counter_PM = 5 and counter_PE = 5 and counter_N = 5 and counter_S = 5 and counter_E = 5 and counter_W = 5) then
    
    
    
    
    
--    counter_PM <= 0;
--    counter_PE <= 0;
--    counter_N  <= 0;
--    counter_S  <= 0;
--    counter_E  <= 0;
--    counter_W  <= 0;
    
     
     
     
       
    
--    end if; 
    
       
     
    clk_count <= 0 ;  
            PM<= '0';
        PE<= '0';
        N <= '0';
        S <= '0';
        E <= '0';
        W <= '0';


case buffer_procedure(0).direction is
      when "000001" =>
       --  if ( i_W_busy='1')then 

                case buffer_procedure(0).channel is 
               
                        when 0 =>  
                          if ( PM_stall_out_EB = '1' and PM_stall_out_DEC ='1')then 
                           
                           
                           i_W_req <= '1';
                           contador <= '1';
                          
                           i_W_pixel <=            PM_in_pixel  ;
                           i_W_x_dest<=            PM_in_x_dest ;
                           i_W_y_dest<=            PM_in_y_dest ;
                           i_W_step  <=            PM_in_step   ;
                           i_W_frame <=            PM_in_frame  ;
                           i_W_x_orig<=            PM_in_x_orig ;
                           i_W_y_orig<=            PM_in_y_orig ;
                           i_W_fb    <=            PM_in_fb     ;
                            
                            PM_SET_ultimo <= '1';
                            end if;
                            
                        when 1 => 
                        if ( PE_stall_out_EB = '1' and PE_stall_out_DEC ='1')then 
                        
                           i_W_req <= '1';
                           contador <= '1';
                                  
                        
                          i_W_pixel <=            PE_in_pixel  ;
                          i_W_x_dest<=            PE_in_x_dest ;
                          i_W_y_dest<=            PE_in_y_dest ;
                          i_W_step  <=            PE_in_step   ;
                          i_W_frame <=            PE_in_frame  ;
                          i_W_x_orig<=            PE_in_x_orig ;
                          i_W_y_orig<=            PE_in_y_orig ;
                          i_W_fb    <=            PE_in_fb     ;
                         
                            
                            PE_SET_ultimo <= '1';
                        end if;
                        
                        when 2 =>                                 
                             if ( N_stall_out_EB = '1' and N_stall_out_DEC ='1')then 
                        
                           i_W_req <= '1';
                           contador <= '1';           
                                        
                                        
                                                                  
                           i_W_pixel <=            N_in_pixel  ;
                           i_W_x_dest<=            N_in_x_dest ;
                           i_W_y_dest<=            N_in_y_dest ;
                           i_W_step  <=            N_in_step   ;
                           i_W_frame <=            N_in_frame  ;
                           i_W_x_orig<=            N_in_x_orig ;
                           i_W_y_orig<=            N_in_y_orig ;
                           i_W_fb    <=            N_in_fb     ;
                            
                            N_SET_ultimo <= '1';
                            end if;
                                                          
                        when 3 => 
                       if ( S_stall_out_EB = '1' and S_stall_out_DEC ='1')then 
                        
                           i_W_req <= '1';
                           contador <= '1';         
                                                                  
                          i_W_pixel <=            S_in_pixel  ; 
                          i_W_x_dest<=            S_in_x_dest ; 
                          i_W_y_dest<=            S_in_y_dest ; 
                          i_W_step  <=            S_in_step   ; 
                          i_W_frame <=            S_in_frame  ; 
                          i_W_x_orig<=            S_in_x_orig ; 
                          i_W_y_orig<=            S_in_y_orig ; 
                          i_W_fb    <=            S_in_fb     ; 
                            
                            S_SET_ultimo <= '1';
                            end if;
                              
                       when 4 =>   
                       if ( E_stall_out_EB = '1' and E_stall_out_DEC ='1')then 
                        
                           i_W_req <= '1';
                           contador <= '1';                              
                                                                 
                          i_W_pixel <=            E_in_pixel  ;
                          i_W_x_dest<=            E_in_x_dest ;
                          i_W_y_dest<=            E_in_y_dest ;
                          i_W_step  <=            E_in_step   ;
                          i_W_frame <=            E_in_frame  ;
                          i_W_x_orig<=            E_in_x_orig ;
                          i_W_y_orig<=            E_in_y_orig ;
                          i_W_fb    <=            E_in_fb     ;
                            
         
                               
                               E_SET_ultimo <= '1';                                   
                       end if;
                       when 5 => 
                           if ( W_stall_out_EB = '1' and W_stall_out_DEC ='1')then 
                        
                           i_W_req <= '1';
                           contador <= '1';                                                 
                                                                 
                         i_W_pixel <=            W_in_pixel  ; 
                         i_W_x_dest<=            W_in_x_dest ; 
                         i_W_y_dest<=            W_in_y_dest ; 
                         i_W_step  <=            W_in_step   ; 
                         i_W_frame <=            W_in_frame  ; 
                         i_W_x_orig<=            W_in_x_orig ; 
                         i_W_y_orig<=            W_in_y_orig ; 
                         i_W_fb    <=            W_in_fb     ; 
                              
                              W_SET_ultimo <= '1';
                                                                        
                          end if;
                                                                  
                    
                    end case; 
                    
                    
  --       end if;
         
      when "100000" =>
         
        -- if ( i_PM_busy='1')then 
             
                    case buffer_procedure(0).channel is 
                         
                          
                        when 0 =>  
                          if ( PM_stall_out_EB = '1' and PM_stall_out_DEC ='1')then 
                           
                           
                           i_PM_req <= '1';
                           contador <= '1';
                           
                           i_PM_pixel <=            PM_in_pixel  ;
                           i_PM_x_dest<=            PM_in_x_dest ;
                           i_PM_y_dest<=            PM_in_y_dest ;
                           i_PM_step  <=            PM_in_step   ;
                           i_PM_frame <=            PM_in_frame  ;
                           i_PM_x_orig<=            PM_in_x_orig ;
                           i_PM_y_orig<=            PM_in_y_orig ;
                           i_PM_fb    <=            PM_in_fb     ;
                            
                            PM_SET_ultimo <= '1';
                           end if;
                        when 1 => 
                    if ( PE_stall_out_EB = '1' and PE_stall_out_DEC ='1')then 
                           
                           
                           i_PM_req <= '1';
                           contador <= '1';  
                        
                          i_PM_pixel <=            PE_in_pixel  ;
                          i_PM_x_dest<=            PE_in_x_dest ;
                          i_PM_y_dest<=            PE_in_y_dest ;
                          i_PM_step  <=            PE_in_step   ;
                          i_PM_frame <=            PE_in_frame  ;
                          i_PM_x_orig<=            PE_in_x_orig ;
                          i_PM_y_orig<=            PE_in_y_orig ;
                          i_PM_fb    <=            PE_in_fb     ;
                          i_PM_fb    <=            PE_in_fb     ;
                           
                            PE_SET_ultimo <= '1';
                            
                          end if;
                        when 2 =>                                 
       if ( N_stall_out_EB = '1' and N_stall_out_DEC ='1')then 
                           
                           
                           i_PM_req <= '1';
                           contador <= '1';                                 
                           i_PM_pixel <=            N_in_pixel  ;
                           i_PM_x_dest<=            N_in_x_dest ;
                           i_PM_y_dest<=            N_in_y_dest ;
                           i_PM_step  <=            N_in_step   ;
                           i_PM_frame <=            N_in_frame  ;
                           i_PM_x_orig<=            N_in_x_orig ;
                           i_PM_y_orig<=            N_in_y_orig ;
                           i_PM_fb    <=            N_in_fb     ;
                            
                            N_SET_ultimo <= '1';
                                end if;                                 
                        when 3 => 
              if (S_stall_out_EB = '1' and S_stall_out_DEC ='1')then 
                           
                           
                           i_PM_req <= '1';
                           contador <= '1';                                            
                                                                  
                          i_PM_pixel <=            S_in_pixel  ; 
                          i_PM_x_dest<=            S_in_x_dest ; 
                          i_PM_y_dest<=            S_in_y_dest ; 
                          i_PM_step  <=            S_in_step   ; 
                          i_PM_frame <=            S_in_frame  ; 
                          i_PM_x_orig<=            S_in_x_orig ; 
                          i_PM_y_orig<=            S_in_y_orig ; 
                          i_PM_fb    <=            S_in_fb     ; 
                          i_PM_fb    <=            S_in_fb     ; 
                          
                           
                           S_SET_ultimo <= '1';
                           end if;  
                       when 4 =>                                 
                    if ( E_stall_out_EB = '1' and E_stall_out_DEC ='1')then 
                           
                           
                           i_PM_req <= '1';
                           contador <= '1';                              
                          i_PM_pixel <=            E_in_pixel  ;
                          i_PM_x_dest<=            E_in_x_dest ;
                          i_PM_y_dest<=            E_in_y_dest ;
                          i_PM_step  <=            E_in_step   ;
                          i_PM_frame <=            E_in_frame  ;
                          i_PM_x_orig<=            E_in_x_orig ;
                          i_PM_y_orig<=            E_in_y_orig ;
                          i_PM_fb    <=            E_in_fb     ;
                          
                               
                               E_SET_ultimo <= '1';
                       end if;                                             
                       when 5 => 
       if ( W_stall_out_EB = '1' and W_stall_out_DEC ='1')then 
                           
                           
                           i_PM_req <= '1';
                           contador <= '1';                 
                                                                 
                         i_PM_pixel <=            W_in_pixel  ; 
                         i_PM_x_dest<=            W_in_x_dest ; 
                         i_PM_y_dest<=            W_in_y_dest ; 
                         i_PM_step  <=            W_in_step   ; 
                         i_PM_frame <=            W_in_frame  ; 
                         i_PM_x_orig<=            W_in_x_orig ; 
                         i_PM_y_orig<=            W_in_y_orig ; 
                         i_PM_fb    <=            W_in_fb     ; 
                        
                             
                          
                          W_SET_ultimo <= '1';                                           
                          
                                  end if;                                
                    
                    end case; 
                    
       --  end if;
 
      when "010000" =>
    
              --  if ( i_PE_busy='1')then 

                    case buffer_procedure(0).channel is 
                        when 0 =>  
                          
                      if ( PM_stall_out_EB = '1' and PM_stall_out_DEC ='1')then    
                      i_PE_req <= '1';
                      contador <= '1';
                          
                          
                           i_PE_pixel <=            PM_in_pixel  ;
                           i_PE_x_dest<=            PM_in_x_dest ;
                           i_PE_y_dest<=            PM_in_y_dest ;
                           i_PE_step  <=            PM_in_step   ;
                           i_PE_frame <=            PM_in_frame  ;
                           i_PE_x_orig<=            PM_in_x_orig ;
                           i_PE_y_orig<=            PM_in_y_orig ;
                           i_PE_fb    <=            PM_in_fb     ;
                            
                           
                            PM_SET_ultimo <= '1';
                            end if;
                            
                        when 1 => 
                   if ( PE_stall_out_EB = '1' and PE_stall_out_DEC ='1')then    
                      i_PE_req <= '1';
                      contador <= '1';
                        
                          i_PE_pixel <=            PE_in_pixel  ;
                          i_PE_x_dest<=            PE_in_x_dest ;
                          i_PE_y_dest<=            PE_in_y_dest ;
                          i_PE_step  <=            PE_in_step   ;
                          i_PE_frame <=            PE_in_frame  ;
                          i_PE_x_orig<=            PE_in_x_orig ;
                          i_PE_y_orig<=            PE_in_y_orig ;
                          i_PE_fb    <=            PE_in_fb     ;
                           
                          
                          PE_SET_ultimo <= '1';
                          END IF;
                        when 2 =>                                 
                     if ( N_stall_out_EB = '1' and N_stall_out_DEC ='1')then    
                      i_PE_req <= '1';
                      contador <= '1';                                
                           i_PE_pixel <=            N_in_pixel  ;
                           i_PE_x_dest<=            N_in_x_dest ;
                           i_PE_y_dest<=            N_in_y_dest ;
                           i_PE_step  <=            N_in_step   ;
                           i_PE_frame <=            N_in_frame  ;
                           i_PE_x_orig<=            N_in_x_orig ;
                           i_PE_y_orig<=            N_in_y_orig ;
                           i_PE_fb    <=            N_in_fb     ;
                       
                                  
                            N_SET_ultimo <= '1';                           
                        END IF;
                        when 3 => 
                      if ( S_stall_out_EB = '1' and S_stall_out_DEC ='1')then    
                      i_PE_req <= '1';
                      contador <= '1';                                        
                                                                  
                          i_PE_pixel <=            S_in_pixel  ; 
                          i_PE_x_dest<=            S_in_x_dest ; 
                          i_PE_y_dest<=            S_in_y_dest ; 
                          i_PE_step  <=            S_in_step   ; 
                          i_PE_frame <=            S_in_frame  ; 
                          i_PE_x_orig<=            S_in_x_orig ; 
                          i_PE_y_orig<=            S_in_y_orig ; 
                          i_PE_fb    <=            S_in_fb     ; 
                          i_PE_fb    <=            S_in_fb     ; 
                          
                          
                          S_SET_ultimo <= '1';
                          END IF;
                       when 4 =>                                 
                     if ( E_stall_out_EB = '1' and E_stall_out_DEC ='1')then    
                      i_PE_req <= '1';
                      contador <= '1';                                       
                          i_PE_pixel <=            E_in_pixel  ;
                          i_PE_x_dest<=            E_in_x_dest ;
                          i_PE_y_dest<=            E_in_y_dest ;
                          i_PE_step  <=            E_in_step   ;
                          i_PE_frame <=            E_in_frame  ;
                          i_PE_x_orig<=            E_in_x_orig ;
                          i_PE_y_orig<=            E_in_y_orig ;
                          i_PE_fb    <=            E_in_fb     ;
                         
                          
                          E_SET_ultimo <= '1';
                        END IF;                                                                  
                       when 5 => 
                   if ( W_stall_out_EB = '1' and W_stall_out_DEC ='1')then    
                      i_PE_req <= '1';
                      contador <= '1';                        
                                                                 
                         i_PE_pixel <=            W_in_pixel  ; 
                         i_PE_x_dest<=            W_in_x_dest ; 
                         i_PE_y_dest<=            W_in_y_dest ; 
                         i_PE_step  <=            W_in_step   ; 
                         i_PE_frame <=            W_in_frame  ; 
                         i_PE_x_orig<=            W_in_x_orig ; 
                         i_PE_y_orig<=            W_in_y_orig ; 
                         i_PE_fb    <=            W_in_fb     ; 
                         
                         
                         W_SET_ultimo <= '1';                                      
                          
                          END IF;                                        
                    
                    end case; 
                    
        -- end if;
      when "001000" =>
     
       --  if ( i_N_busy='1')then              
      case buffer_procedure(0).channel is 
                        when 0 =>  
                     If( PM_stall_out_EB = '1' and PM_stall_out_DEC ='1')then    
                      i_N_req <= '1';
                      contador <= '1';
                           i_N_pixel <=            PM_in_pixel  ;
                           i_N_x_dest<=            PM_in_x_dest ;
                           i_N_y_dest<=            PM_in_y_dest ;
                           i_N_step  <=            PM_in_step   ;
                           i_N_frame <=            PM_in_frame  ;
                           i_N_x_orig<=            PM_in_x_orig ;
                           i_N_y_orig<=            PM_in_y_orig ;
                           i_N_fb    <=            PM_in_fb     ;
                          
                            
                            PM_SET_ultimo <= '1';
                            END IF;
                        when 1 => 
                          If( PE_stall_out_EB = '1' and PE_stall_out_DEC ='1')then            
                                              i_N_req <= '1';
                      contador <= '1';
                          i_N_pixel <=            PE_in_pixel  ;
                          i_N_x_dest<=            PE_in_x_dest ;
                          i_N_y_dest<=            PE_in_y_dest ;
                          i_N_step  <=            PE_in_step   ;
                          i_N_frame <=            PE_in_frame  ;
                          i_N_x_orig<=            PE_in_x_orig ;
                          i_N_y_orig<=            PE_in_y_orig ;
                          i_N_fb    <=            PE_in_fb     ;
                          
                          
                          PE_SET_ultimo <= '1';
                            end if;
                        when 2 =>                                 
                            If( N_stall_out_EB = '1' and N_stall_out_DEC ='1')then                         
                                                    i_N_req <= '1';
                      contador <= '1';                                    
                           i_N_pixel <=            N_in_pixel  ;
                           i_N_x_dest<=            N_in_x_dest ;
                           i_N_y_dest<=            N_in_y_dest ;
                           i_N_step  <=            N_in_step   ;
                           i_N_frame <=            N_in_frame  ;
                           i_N_x_orig<=            N_in_x_orig ;
                           i_N_y_orig<=            N_in_y_orig ;
                           i_N_fb    <=            N_in_fb     ;
                            
                                                            
                            N_SET_ultimo <= '1';         
                            end if;                          
                        when 3 => 
                              If( S_stall_out_EB = '1' and S_stall_out_DEC ='1')then                         
                                                                       i_N_req <= '1';
                      contador <= '1';                 
                          i_N_pixel <=            S_in_pixel  ; 
                          i_N_x_dest<=            S_in_x_dest ; 
                          i_N_y_dest<=            S_in_y_dest ; 
                          i_N_step  <=            S_in_step   ; 
                          i_N_frame <=            S_in_frame  ; 
                          i_N_x_orig<=            S_in_x_orig ; 
                          i_N_y_orig<=            S_in_y_orig ; 
                          i_N_fb    <=            S_in_fb     ; 
                         
                           
                           S_SET_ultimo <= '1';
                           end if; 
                       when 4 =>                                 
                               If( E_stall_out_EB = '1' and E_stall_out_DEC ='1')then           
                                                      i_N_req <= '1';
                      contador <= '1';                                 
                          i_N_pixel <=            E_in_pixel  ;
                          i_N_x_dest<=            E_in_x_dest ;
                          i_N_y_dest<=            E_in_y_dest ;
                          i_N_step  <=            E_in_step   ;
                          i_N_frame <=            E_in_frame  ;
                          i_N_x_orig<=            E_in_x_orig ;
                          i_N_y_orig<=            E_in_y_orig ;
                          i_N_fb    <=            E_in_fb     ;
                        
                          
                          E_SET_ultimo <= '1';
                            end if;                                           
                       when 5 => 
                            If( W_stall_out_EB = '1' and W_stall_out_DEC ='1')then                          
                                                            i_N_req <= '1';
                      contador <= '1';                           
                         i_N_pixel <=            W_in_pixel  ; 
                         i_N_x_dest<=            W_in_x_dest ; 
                         i_N_y_dest<=            W_in_y_dest ; 
                         i_N_step  <=            W_in_step   ; 
                         i_N_frame <=            W_in_frame  ; 
                         i_N_x_orig<=            W_in_x_orig ; 
                         i_N_y_orig<=            W_in_y_orig ; 
                         i_N_fb    <=            W_in_fb     ; 
                        
                         
                         W_SET_ultimo <= '1';
                                                                        
                          
                                                                  
                      end if;
                    end case; 
                    
       --  end if;
      when "000100" =>
        
        -- if ( i_S_busy = '1') then 
        
            case buffer_procedure(0).channel is 
                        when 0 =>  
                               If( PM_stall_out_EB = '1' and PM_stall_out_DEC ='1')then             
                                               i_S_req <= '1';
            contador <= '1';          
                                                               
                           i_S_pixel <=            PM_in_pixel  ;
                           i_S_x_dest<=            PM_in_x_dest ;
                           i_S_y_dest<=            PM_in_y_dest ;
                           i_S_step  <=            PM_in_step   ;
                           i_S_frame <=            PM_in_frame  ;
                           i_S_x_orig<=            PM_in_x_orig ;
                           i_S_y_orig<=            PM_in_y_orig ;
                           i_S_fb    <=            PM_in_fb     ;
                           
                           
                           PM_SET_ultimo <= '1';
                            end if; 
                        when 1 => 
                          If( PE_stall_out_EB = '1' and PE_stall_out_DEC ='1')then       
                               i_S_req <= '1';
            contador <= '1';          
                          i_S_pixel <=            PE_in_pixel  ;
                          i_S_x_dest<=            PE_in_x_dest ;
                          i_S_y_dest<=            PE_in_y_dest ;
                          i_S_step  <=            PE_in_step   ;
                          i_S_frame <=            PE_in_frame  ;
                          i_S_x_orig<=            PE_in_x_orig ;
                          i_S_y_orig<=            PE_in_y_orig ;
                          i_S_fb    <=            PE_in_fb     ;
                           
                           
                           PE_SET_ultimo <= '1';
                           end if;
                        when 2 =>                                 
                                        
                          If( N_stall_out_EB = '1' and N_stall_out_DEC ='1')then              
                                               i_S_req <= '1';
            contador <= '1';                                    
                           i_S_pixel <=            N_in_pixel  ;
                           i_S_x_dest<=            N_in_x_dest ;
                           i_S_y_dest<=            N_in_y_dest ;
                           i_S_step  <=            N_in_step   ;
                           i_S_frame <=            N_in_frame  ;
                           i_S_x_orig<=            N_in_x_orig ;
                           i_S_y_orig<=            N_in_y_orig ;
                           i_S_fb    <=            N_in_fb     ;
                                
                            N_SET_ultimo <= '1'    ;    
                            end if;                        
                        when 3 => 
                                 If( S_stall_out_EB = '1' and S_stall_out_DEC ='1')then                    
                                               i_S_req <= '1';
            contador <= '1';                                    
                          i_S_pixel <=            S_in_pixel  ; 
                          i_S_x_dest<=            S_in_x_dest ; 
                          i_S_y_dest<=            S_in_y_dest ; 
                          i_S_step  <=            S_in_step   ; 
                          i_S_frame <=            S_in_frame  ; 
                          i_S_x_orig<=            S_in_x_orig ; 
                          i_S_y_orig<=            S_in_y_orig ; 
                          i_S_fb    <=            S_in_fb     ; 
                       
                           
                           S_SET_ultimo <= '1';
                         end if;   
                       when 4 =>   
                        If( E_stall_out_EB = '1' and E_stall_out_DEC ='1')then                               
                                               i_S_req <= '1';
            contador <= '1';                                   
                          i_S_pixel <=            E_in_pixel  ;
                          i_S_x_dest<=            E_in_x_dest ;
                          i_S_y_dest<=            E_in_y_dest ;
                          i_S_step  <=            E_in_step   ;
                          i_S_frame <=            E_in_frame  ;
                          i_S_x_orig<=            E_in_x_orig ;
                          i_S_y_orig<=            E_in_y_orig ;
                          i_S_fb    <=            E_in_fb     ;
                           
                               
                               E_SET_ultimo <= '1';
                            end if;                                    
                       when 5 => 
                        If( W_stall_out_EB = '1' and W_stall_out_DEC ='1')then 
                                                  
                         i_S_req <= '1';
                         contador <= '1';                                   
                         i_S_pixel <=            W_in_pixel  ; 
                         i_S_x_dest<=            W_in_x_dest ; 
                         i_S_y_dest<=            W_in_y_dest ; 
                         i_S_step  <=            W_in_step   ; 
                         i_S_frame <=            W_in_frame  ; 
                         i_S_x_orig<=            W_in_x_orig ; 
                         i_S_y_orig<=            W_in_y_orig ; 
                         i_S_fb    <=            W_in_fb     ; 
                              
                              
                              W_SET_ultimo <= '1';
                                                                        
                            end if;
                                                                  
                    
                    end case; 
                    
        -- end if;
      when "000010" =>
         
         --if ( i_E_busy='1')then 
       
                case buffer_procedure(0).channel is 
                        when 0 =>  
                         If( PM_stall_out_EB = '1' and PM_stall_out_DEC ='1')then 
                                       contador <= '1';
                  i_E_req <= '1';       
                           i_E_pixel <=            PM_in_pixel  ;
                           i_E_x_dest<=            PM_in_x_dest ;
                           i_E_y_dest<=            PM_in_y_dest ;
                           i_E_step  <=            PM_in_step   ;
                           i_E_frame <=            PM_in_frame  ;
                           i_E_x_orig<=            PM_in_x_orig ;
                           i_E_y_orig<=            PM_in_y_orig ;
                           i_E_fb    <=            PM_in_fb     ;
                       
                                               
                           PM_SET_ultimo <= '1';
                            end if; 
                        when 1 => 
                                   If( PE_stall_out_EB = '1' and PE_stall_out_DEC ='1')then 
                                     contador <= '1';
                  i_E_req <= '1';       
                          i_E_pixel <=            PE_in_pixel  ;
                          i_E_x_dest<=            PE_in_x_dest ;
                          i_E_y_dest<=            PE_in_y_dest ;
                          i_E_step  <=            PE_in_step   ;
                          i_E_frame <=            PE_in_frame  ;
                          i_E_x_orig<=            PE_in_x_orig ;
                          i_E_y_orig<=            PE_in_y_orig ;
                          i_E_fb    <=            PE_in_fb     ;
                          
                          
                          PE_SET_ultimo <= '1';
                          end if;
                        when 2 =>   
                         If( N_stall_out_EB = '1' and N_stall_out_DEC ='1')then                               
                                          contador <= '1';
                  i_E_req <= '1';                                            
                           i_E_pixel <=            N_in_pixel  ;
                           i_E_x_dest<=            N_in_x_dest ;
                           i_E_y_dest<=            N_in_y_dest ;
                           i_E_step  <=            N_in_step   ;
                           i_E_frame <=            N_in_frame  ;
                           i_E_x_orig<=            N_in_x_orig ;
                           i_E_y_orig<=            N_in_y_orig ;
                           i_E_fb    <=            N_in_fb     ;
                            
                            
                            N_SET_ultimo <= '1';
                            
                            end if;                                                                     
                        when 3 => 
                               If( S_stall_out_EB = '1' and S_stall_out_DEC ='1')then                      
                                         contador <= '1';
                  i_E_req <= '1';                                             
                          i_E_pixel <=            S_in_pixel  ; 
                          i_E_x_dest<=            S_in_x_dest ; 
                          i_E_y_dest<=            S_in_y_dest ; 
                          i_E_step  <=            S_in_step   ; 
                          i_E_frame <=            S_in_frame  ; 
                          i_E_x_orig<=            S_in_x_orig ; 
                          i_E_y_orig<=            S_in_y_orig ; 
                          i_E_fb    <=            S_in_fb     ; 
                          i_E_fb    <=            S_in_fb     ; 
                          
                          
                          S_SET_ultimo <= '1';
                         end if;
                       when 4 =>   
                        If( E_stall_out_EB = '1' and E_stall_out_DEC ='1')then                               
                                        contador <= '1';
                  i_E_req <= '1';                                             
                          i_E_pixel <=            E_in_pixel  ;
                          i_E_x_dest<=            E_in_x_dest ;
                          i_E_y_dest<=            E_in_y_dest ;
                          i_E_step  <=            E_in_step   ;
                          i_E_frame <=            E_in_frame  ;
                          i_E_x_orig<=            E_in_x_orig ;
                          i_E_y_orig<=            E_in_y_orig ;
                          i_E_fb    <=            E_in_fb     ;
                            
                              
                               E_SET_ultimo <= '1';
                                 
                           end if;                                     
                       when 5 => 
                       If( W_stall_out_EB = '1' and W_stall_out_DEC ='1')then                   
                                        contador <= '1';
                  i_E_req <= '1';                                             
                         i_E_pixel <=            W_in_pixel  ; 
                         i_E_x_dest<=            W_in_x_dest ; 
                         i_E_y_dest<=            W_in_y_dest ; 
                         i_E_step  <=            W_in_step   ; 
                         i_E_frame <=            W_in_frame  ; 
                         i_E_x_orig<=            W_in_x_orig ; 
                         i_E_y_orig<=            W_in_y_orig ; 
                         i_E_fb    <=            W_in_fb     ; 
                        
                          
                         W_SET_ultimo <= '1';                                       
                          
                                                                  
                      end if;
                    end case; 
                    
       --  end if;
         
         
         
     when others =>
   end case;
   

   when '1' => 
   
   case contador_two is 
   
  when xxx =>
         if ( i_PM_ack = '1') then 
          PM<= '1' ;
           i_PM_req<='0';      
            if (PM_SET_ultimo = '1') then
            PM_stall_in_EB <= '1'; 
            PM_stall_in_DEC <= '1';
            contador_two <= yyy;       
            end if;
            if (PE_SET_ultimo = '1') then
            PE_stall_in_EB <= '1'; 
            PE_stall_in_DEC <= '1';
            contador_two <= yyy;      
            end if;
            if (N_SET_ultimo = '1') then
            N_stall_in_EB <= '1'; 
            N_stall_in_DEC <= '1';
            contador_two <= yyy;       
            end if;
            if (S_SET_ultimo = '1') then
            S_stall_in_EB <= '1'; 
            S_stall_in_DEC <= '1';
            contador_two <= yyy;       
            end if;
            if (E_SET_ultimo = '1') then
            E_stall_in_EB <= '1'; 
            E_stall_in_DEC <= '1';
            contador_two <= yyy;      
            end if;
            if (W_SET_ultimo = '1') then
            W_stall_in_EB <= '1'; 
            W_stall_in_DEC <= '1';
            contador_two <= yyy;      
            end if; 

        end if;
        
         if ( i_PE_ack = '1') then 
        PE<= '1' ;
        i_PE_req<='0';

            if (PM_SET_ultimo = '1') then
            PM_stall_in_EB <= '1'; 
            PM_stall_in_DEC <= '1';
             contador_two <= yyy;        
            end if;
            if (PE_SET_ultimo = '1') then
            PE_stall_in_EB <= '1'; 
            PE_stall_in_DEC <= '1';
              contador_two <= yyy;     
            end if;
            if (N_SET_ultimo = '1') then
            N_stall_in_EB <= '1'; 
            N_stall_in_DEC <= '1';
             contador_two <= yyy;       
            end if;
            if (S_SET_ultimo = '1') then
            S_stall_in_EB <= '1'; 
            S_stall_in_DEC <= '1';
             contador_two <= yyy;    
            end if;
            if (E_SET_ultimo = '1') then
            E_stall_in_EB <= '1'; 
            E_stall_in_DEC <= '1';
             contador_two <= yyy;  
            end if;
            if (W_SET_ultimo = '1') then
            W_stall_in_EB <= '1'; 
            W_stall_in_DEC <= '1';
             contador_two <= yyy;       
            end if; 

        end if;

         if ( i_N_ack = '1') then 
        N<= '1' ;
        i_N_req<='0';
            if (PM_SET_ultimo = '1') then
            PM_stall_in_EB <= '1'; 
            PM_stall_in_DEC <= '1';
              contador_two <= yyy;      
            end if;
            if (PE_SET_ultimo = '1') then
            PE_stall_in_EB <= '1'; 
            PE_stall_in_DEC <= '1';
             contador_two <= yyy;      
            end if;
            if (N_SET_ultimo = '1') then
            N_stall_in_EB <= '1'; 
            N_stall_in_DEC <= '1';
             contador_two <= yyy;        
            end if;
            if (S_SET_ultimo = '1') then
            S_stall_in_EB <= '1'; 
            S_stall_in_DEC <= '1';
             contador_two <= yyy;       
            end if;
            if (E_SET_ultimo = '1') then
            E_stall_in_EB <= '1'; 
            E_stall_in_DEC <= '1';
             contador_two <= yyy;     
            end if;
            if (W_SET_ultimo = '1') then
            W_stall_in_EB <= '1'; 
            W_stall_in_DEC <= '1';
              contador_two <= yyy;       
         
            end if; 

        end if;
    
         if ( i_S_ack = '1') then 
        S<= '1' ;
        i_S_req<='0';
            if (PM_SET_ultimo = '1') then
            PM_stall_in_EB <= '1'; 
            PM_stall_in_DEC <= '1';
             contador_two <= yyy;       
            end if;
            if (PE_SET_ultimo = '1') then
            PE_stall_in_EB <= '1'; 
            PE_stall_in_DEC <= '1';
             contador_two <= yyy;       
            end if;
            if (N_SET_ultimo = '1') then
            N_stall_in_EB <= '1'; 
            N_stall_in_DEC <= '1';
             contador_two <= yyy;      
            end if;
            if (S_SET_ultimo = '1') then
            S_stall_in_EB <= '1'; 
            S_stall_in_DEC <= '1';
             contador_two <= yyy;        
            end if;
            if (E_SET_ultimo = '1') then
            E_stall_in_EB <= '1'; 
            E_stall_in_DEC <= '1';
              contador_two <= yyy;       
            end if;
            if (W_SET_ultimo = '1') then
            W_stall_in_EB <= '1'; 
            W_stall_in_DEC <= '1';
             contador_two <= yyy;        
         
            end if; 

        end if;
               
         if ( i_E_ack = '1') then 
          E<= '1' ;
           i_E_req<='0';
            if (PM_SET_ultimo = '1') then
            PM_stall_in_EB <= '1'; 
            PM_stall_in_DEC <= '1';
            contador_two <= yyy;      
            end if;
            if (PE_SET_ultimo = '1') then
            PE_stall_in_EB <= '1'; 
            PE_stall_in_DEC <= '1';
            contador_two <= yyy;        
         
            end if;
            if (N_SET_ultimo = '1') then
            N_stall_in_EB <= '1'; 
            N_stall_in_DEC <= '1';
             contador_two <= yyy;        
         
            end if;
            if (S_SET_ultimo = '1') then
            S_stall_in_EB <= '1'; 
            S_stall_in_DEC <= '1';
              contador_two <= yyy;      
         
            end if;
            if (E_SET_ultimo = '1') then
            E_stall_in_EB <= '1'; 
            E_stall_in_DEC <= '1';
            contador_two <= yyy;
        
            end if;
            if (W_SET_ultimo = '1') then
            W_stall_in_EB <= '1'; 
            W_stall_in_DEC <= '1';
            contador_two <= yyy;     
        
            end if; 

        end if;
        
         if ( i_W_ack = '1') then 
          W<= '1' ;
          i_W_req<='0';

            if (PM_SET_ultimo = '1') then
            PM_stall_in_EB <= '1'; 
            PM_stall_in_DEC <= '1';
             contador_two <= yyy;        
            end if;
            if (PE_SET_ultimo = '1') then
            PE_stall_in_EB <= '1'; 
            PE_stall_in_DEC <= '1';
             contador_two <= yyy;        
            end if;
            if (N_SET_ultimo = '1') then
            N_stall_in_EB <= '1'; 
            N_stall_in_DEC <= '1';
             contador_two <= yyy;        
            end if;
            if (S_SET_ultimo = '1') then
            S_stall_in_EB <= '1'; 
            S_stall_in_DEC <= '1';
             contador_two <= yyy;        
            end if;
            if (E_SET_ultimo = '1') then
            E_stall_in_EB <= '1'; 
            E_stall_in_DEC <= '1';
             contador_two <= yyy;        
            end if;
            if (W_SET_ultimo = '1') then
            W_stall_in_EB <= '1'; 
            W_stall_in_DEC <= '1';
              contador_two <= yyy;  
            end if; 

        end if;
        
        
        clk_count<= clk_count+1;
        
        if (clk_count >=10) then
        if(PM ='0'and PE='0'and N='0'and E='0'and S='0'and W='0' )then  
               

        i_PM_req<='0';
        i_PE_req<='0'; 
        i_N_req<='0';
        i_S_req<='0'; 
        i_E_req<='0';
        i_W_req<='0'; 
        clk_count <= 0 ;   
        
        contador_two <= xxx ;       
        contador <= '0';
        
        
        PM<= '0';
        PE<= '0';
        N <= '0';
        S <= '0';
        E <= '0';
        W <= '0';
        
        end if;
        
        end if;
        
  when yyy =>
    
     clk_count <= 0 ;
     contador <= '1';
     contador_two <= xxx;
     PM_stall_in_EB <=  '0'; 
PM_stall_in_DEC <= '0';
PE_stall_in_EB <=  '0'; 
PE_stall_in_DEC <= '0';
N_stall_in_EB <=   '0'; 
N_stall_in_DEC <=  '0';
S_stall_in_EB <=   '0'; 
S_stall_in_DEC <=  '0';
E_stall_in_EB <=   '0'; 
E_stall_in_DEC <=  '0';
W_stall_in_EB <=   '0'; 
W_stall_in_DEC <=  '0';
     
       contador <= '0';
     
     
  when zzz =>
  
  contador_two <= xxx; 

  

     

        when others =>
        
        
        
        
        end case;
        
        
        
        
        
        
        
        
        
        
        
    when others => 
    
    end case ; 
    
    
    end if;
    
    
    
    end process;
 end architecture rtl;














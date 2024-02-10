library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use ieee.std_logic_unsigned.all;



entity S_elastic_buffer is
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
 
clk                 : in std_logic;
rst                 : in std_logic;
 
--S_ready_CF_EB       : in       std_logic;
S_valid_EB_DEC      : out      std_logic ; -- indica que existe mensagem pronta para ser decodificada pelo decoder
S_ready_DEC_EB      : in       std_logic; --- decoder indica pro EB que pode receber msg

S_out_new_EB_DEC    : out      std_logic; --- indica que mensagem foi enviada do decoder para o EB
S_ready_EB_CF       : out      std_logic ; -- informa ao CF que o FIFO est? pronto para receber mais uma mensagem. 
 --- variavel indica que existe mensagem no controlador de fluxo no outro router pronta para ler enqueu pelo EB
S_stall_in_EB       : in       std_logic := '0'; --- indica que o processo deve ser stall para que exista sincronia dos elementos -> no CF
S_stall_out_EB      : inout    std_logic := '0'; -- indica que o '''
 
S_data_in_pixel     : in std_logic_vector(pix_depth-1 downto 0)  ;
S_data_in_x_dest    : in std_logic_vector(img_width-1 downto 0)  ;
S_data_in_y_dest    : in std_logic_vector(img_height-1 downto 0) ;
S_data_in_step      : in std_logic_vector(n_steps-1 downto 0)    ;
S_data_in_frame     : in std_logic_vector(n_frames-1 downto 0)   ;
S_data_in_x_orig    : in std_logic_vector(img_width-1 downto 0)  ;
S_data_in_y_orig    : in std_logic_vector(img_height-1 downto 0) ;
S_data_in_fb        : in std_logic;

S_req               : in std_logic := '0';            
S_ack               : out std_logic := '0';             


 
S_data_out_pixel     : out std_logic_vector(pix_depth-1 downto 0);    
S_data_out_x_dest    : out std_logic_vector(img_width-1 downto 0);    
S_data_out_y_dest    : out std_logic_vector(img_height-1 downto 0);   
S_data_out_step      : out std_logic_vector(n_steps-1 downto 0);      
S_data_out_frame     : out std_logic_vector(n_frames-1 downto 0);     
S_data_out_x_orig    : out std_logic_vector(img_width-1 downto 0);    
S_data_out_y_orig    : out std_logic_vector(img_height-1 downto 0);   
S_data_out_fb        : out std_logic  
                                
 
 
 );





end S_elastic_buffer;

architecture Behavioral of S_elastic_buffer is

 type t_FIFO_pix_depth      is array (0 to buffer_length-1) of std_logic_vector(pix_depth-1 downto 0);
 type t_FIFO_img_width      is array (0 to buffer_length-1) of std_logic_vector(img_width-1 downto 0);
 type t_FIFO_img_height     is array (0 to buffer_length-1) of std_logic_vector(img_height-1 downto 0);
 type t_FIFO_n_steps        is array (0 to buffer_length-1) of std_logic_vector(n_steps-1 downto 0);
 type t_FIFO_n_frames       is array (0 to buffer_length-1) of std_logic_vector(n_frames-1 downto 0);
 type t_FIFO_x_orig         is array (0 to buffer_length-1) of std_logic_vector(img_width-1 downto 0);
 type t_FIFO_y_orig         is array (0 to buffer_length-1) of std_logic_vector(img_height-1 downto 0);
 type t_FIFO_fb             is array (0 to buffer_length-1) of std_logic;

signal write : integer range 0 to buffer_length-1 := 0;-- range 0 to buffer_length-1 := 0;
signal read : integer range 0 to buffer_length-1 := 0;-- range 0 to buffer_length-1  := 0;

signal full :  std_logic_vector(0 to buffer_length-1) := (others => '0');

signal S_data_pix_depth  : t_FIFO_pix_depth   ;--  := (others => (others => '0'));
signal S_data_img_width  : t_FIFO_img_width   ;--  := (others => (others => '0'));
signal S_data_img_height : t_FIFO_img_height  ;--  := (others => (others => '0'));
signal S_data_n_steps    : t_FIFO_n_steps     ;--  := (others => (others => '0'));
signal S_data_n_frames   : t_FIFO_n_frames    ;--  := (others => (others => '0'));
signal S_data_x_orig     : t_FIFO_x_orig      ;--  := (others => (others => '0'));                         
signal S_data_y_orig     : t_FIFO_y_orig      ;--  := (others => (others => '0'));                     
signal S_data_fb         : t_FIFO_fb          ;--  := (others => '0');        
signal S_stall_s         : std_logic := '0';
--signal S_ready_EB_CF         : std_logic := '0';
signal state : std_logic := '0';
signal state_write : std_logic := '0' ;


type clean_EB is
record
    
t_S_data_pix_depth  : t_FIFO_pix_depth   ;
t_S_data_img_width  : t_FIFO_img_width   ;
t_S_data_img_height : t_FIFO_img_height  ;
t_S_data_n_steps    : t_FIFO_n_steps     ;
t_S_data_n_frames   : t_FIFO_n_frames    ;
t_S_data_x_orig     : t_FIFO_x_orig      ;
t_S_data_y_orig     : t_FIFO_y_orig      ;
t_S_data_fb         : t_FIFO_fb          ;
      
   -- channel : integer range 0 to 6;    
end record;


type clean is array(0 to (buffer_length-1)) of clean_EB;

   
    constant  constant_clean : clean_EB := (   t_S_data_pix_depth   =>(others => (others => '0')),
                                               t_S_data_img_width  => (others => (others => '0')),
                                               t_S_data_img_height => (others => (others => '0')),
                                               t_S_data_n_steps    => (others => (others => '0')),
                                               t_S_data_n_frames   => (others => (others => '0')),
                                               t_S_data_x_orig     => (others => (others => '0')),
                                               t_S_data_y_orig     => (others => (others => '0')),
                                               t_S_data_fb         => (others => '0')   );       
      









begin


S_valid_EB_DEC <= or(full) ;

S_ready_EB_CF <= or(not(full));





 process(clk,rst) 
 begin 
    
    if (rst='1' ) then 
    
    
S_data_pix_depth   <= (others => (others => '0'));
S_data_img_width   <= (others => (others => '0'));
S_data_img_height  <= (others => (others => '0'));
S_data_n_steps     <= (others => (others => '0'));
S_data_n_frames    <= (others => (others => '0'));
S_data_x_orig      <= (others => (others => '0'));
S_data_y_orig      <= (others => (others => '0'));
S_data_fb          <= (others => '0')       ;



 S_data_out_pixel  <= (others=> '0');
 S_data_out_x_dest <= (others=> '0');
 S_data_out_y_dest <= (others=> '0');
 S_data_out_step   <= (others=> '0');
 S_data_out_frame  <= (others=> '0');
 S_data_out_x_orig <= (others=> '0');
 S_data_out_y_orig <= (others=> '0');
 S_data_out_fb     <= '0';

    
    
    
    
    
    
    elsif (rising_edge(clk)) then 
            case state_write is 
      
            when '0' =>
            if (  full(write) = '0') then        -- S_valid_CF_EB ='1'       
               
               if (S_req = '1') then 
               S_ack <= '1';
               full(write) <=  '1';           
               state_write <= '1';
               S_data_pix_depth(write) <= S_data_in_pixel  ;
               S_data_img_width(write) <= S_data_in_x_dest ;
               S_data_img_height(write)<= S_data_in_y_dest ;
               S_data_n_steps(write)   <= S_data_in_step   ;
               S_data_n_frames(write)  <= S_data_in_frame  ;
               S_data_x_orig(write)    <= S_data_in_x_orig ;
               S_data_y_orig(write)    <= S_data_in_y_orig ;
               S_data_fb(write)        <= S_data_in_fb     ;
             
             
                if( write < buffer_length-1) then              
                write <= write + 1;
                elsif( write = buffer_length-1) then              
                write <= 0;
                end if ;

     
            end if;
            end if;
            
            
           when '1' => 
           state_write <= '0';
           S_ack<='0';
           
                   
           
           
           
                      
           when others =>
           
           end case;  
  ---------------------------------------------------------------------------------
  ---------------------------------------------------------------------------------
  --------------------------------------------------------------------------------
  -------------------------------------------------------------------------------          
--   case S_stall_s is
--   when '0' =>
-- if (  S_stall_in_EB ='0' )  then 
        
                    case state is 
                    when '0' =>                              
                    if (S_ready_DEC_EB = '1' and full(read) = '1') then 
                                 state<='1';
                                S_stall_out_EB <= '1';
                              
                                if ( read <  buffer_length-1 ) then 
                                full(read) <= '0';
                                read<= read + 1;
                                S_out_new_EB_DEC<= '1';

                                elsif ( read =  buffer_length-1 ) then 
                                full(read) <= '0';
                                read<= 0;  
                                 S_out_new_EB_DEC<= '1';          
                                end if;
                                
                                
                                S_data_out_pixel      <=   S_data_pix_depth((read));
                                S_data_out_x_dest     <=   S_data_img_width(read)  ;
                                S_data_out_y_dest     <=   S_data_img_height(read) ;
                                S_data_out_step       <=   S_data_n_steps(read)    ; 
                                S_data_out_frame      <=   S_data_n_frames(read)   ;
                                S_data_out_x_orig     <=   S_data_x_orig(read)     ; 
                                S_data_out_y_orig     <=   S_data_y_orig(read)     ;
                                S_data_out_fb         <=   S_data_fb(read)         ;

                    
                    
                    

  end if;  
           
            when '1' =>
                  
                  
                  S_out_new_EB_DEC<= '0'; 
                  if( S_stall_in_EB = '1') then 
                  S_stall_out_EB<= '0';   
                  state<='0';
                  
                  end if;
                             
            when others =>
            
              
          end case;

       
       
       

       end if;     
            
       
             



 


 end process;

end Behavioral;